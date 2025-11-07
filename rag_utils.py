"""Utilities for document ingestion and retrieval-augmented generation."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

import chardet

try:  # pragma: no cover - handled gracefully at runtime
    import faiss
except ImportError as exc:  # pragma: no cover - surfaced in UI instead
    faiss = None
    FAISS_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when FAISS is available
    FAISS_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import tiktoken
except ImportError:  # pragma: no cover - fall back to heuristic token estimate
    tiktoken = None

try:  # pragma: no cover - optional dependency
    from docx import Document
except ImportError as exc:  # pragma: no cover - surfaced during ingestion
    Document = None
    DOCX_IMPORT_ERROR = exc
else:  # pragma: no cover
    DOCX_IMPORT_ERROR = None

from pypdf import PdfReader


TEXT_EXTENSIONS = {"txt", "md"}
CSV_EXTENSIONS = {"csv"}
EXCEL_EXTENSIONS = {"xlsx", "xls"}
PDF_EXTENSIONS = {"pdf"}
DOCX_EXTENSIONS = {"docx"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | CSV_EXTENSIONS | EXCEL_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS


@dataclass
class DocumentChunk:
    text: str
    meta: Dict[str, Any]


def _clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[str]:
    """Chunk a long string into overlapping segments."""
    if not text:
        return []

    normalized = _clean_text(text)
    if len(normalized) < 50:
        return []

    chunks: List[str] = []
    start = 0
    step = max_chars - overlap
    if step <= 0:
        step = max_chars

    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        chunk = normalized[start:end].strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start += step

    return chunks


def _estimate_tokens(text: str) -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:  # pragma: no cover - fallback used if encoding fails
            pass
    return max(1, math.ceil(len(text) / 4))


def _detect_encoding(data: bytes) -> str:
    guess = chardet.detect(data)
    encoding = guess.get("encoding") or "utf-8"
    return encoding


def _bytes_to_text(data: bytes) -> str:
    encoding = _detect_encoding(data)
    try:
        return data.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        return data.decode("utf-8", errors="ignore")


def _ingest_text_document(
    name: str,
    data: bytes,
    file_type: str,
    *,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    text_content = _bytes_to_text(data)
    chunks = [
        DocumentChunk(text=chunk, meta={"source": name, "type": file_type})
        for chunk in chunk_text(text_content, max_chars=max_chars, overlap=overlap)
    ]
    summary = {
        "name": name,
        "type": file_type,
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, []


def _ingest_csv(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    encoding = _detect_encoding(data)
    warnings: List[str] = []
    try:
        df = pd.read_csv(io.BytesIO(data), encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8", errors="ignore")
    except Exception as exc:
        return [], {}, [f"Erreur lors de la lecture CSV {name}: {exc}"]

    if df.empty:
        warnings.append(f"Le fichier CSV {name} est vide.")
        summary = {
            "name": name,
            "type": "csv",
            "size_bytes": len(data),
            "chunk_count": 0,
            "token_estimate": 0,
        }
        return [], summary, warnings

    csv_text = df.to_csv(index=False)
    lines = csv_text.splitlines()

    chunks: List[DocumentChunk] = []
    line_start = 1
    buffer: List[str] = []
    char_count = 0
    max_chars_local = max_chars
    overlap_local = overlap

    for i, line in enumerate(lines, start=1):
        line_with_newline = line + "\n"
        if char_count + len(line_with_newline) > max_chars_local and buffer:
            chunk_text_value = "".join(buffer)
            for part in chunk_text(chunk_text_value, max_chars=max_chars_local, overlap=overlap_local):
                chunks.append(
                    DocumentChunk(
                        text=part,
                        meta={
                            "source": name,
                            "type": "csv",
                            "row_range": f"{line_start}-{i-1}",
                        },
                    )
                )
            buffer = []
            char_count = 0
            line_start = i

        buffer.append(line_with_newline)
        char_count += len(line_with_newline)

    if buffer:
        chunk_text_value = "".join(buffer)
        for part in chunk_text(chunk_text_value, max_chars=max_chars_local, overlap=overlap_local):
            chunks.append(
                DocumentChunk(
                    text=part,
                    meta={
                        "source": name,
                        "type": "csv",
                        "row_range": f"{line_start}-{len(lines)}",
                    },
                )
            )

    summary = {
        "name": name,
        "type": "csv",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, warnings


def _ingest_excel(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    except Exception as exc:
        return [], {}, [f"Erreur lors de la lecture Excel {name}: {exc}"]

    chunks: List[DocumentChunk] = []
    for sheet_name, df in sheets.items():
        if df.empty:
            warnings.append(f"La feuille '{sheet_name}' dans {name} est vide.")
            continue
        text = df.to_csv(index=False)
        for part in chunk_text(text, max_chars=max_chars, overlap=overlap):
            chunks.append(
                DocumentChunk(
                    text=part,
                    meta={"source": name, "type": "excel", "sheet": sheet_name},
                )
            )

    summary = {
        "name": name,
        "type": "excel",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, warnings


def _ingest_pdf(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    pdf_reader = PdfReader(io.BytesIO(data))
    chunks: List[DocumentChunk] = []
    warnings: List[str] = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"Extraction impossible page {page_number} ({name}) : {exc}")
            continue
        cleaned = _clean_text(text)
        if not cleaned:
            warnings.append(f"Page {page_number} du PDF {name} semble ne contenir aucun texte.")
            continue
        for chunk in chunk_text(cleaned, max_chars=max_chars, overlap=overlap):
            chunks.append(
                DocumentChunk(text=chunk, meta={"source": name, "type": "pdf", "page": page_number})
            )

    summary = {
        "name": name,
        "type": "pdf",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, warnings


def _ingest_docx(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    if Document is None:
        return [], {}, [f"Le support DOCX est indisponible : {DOCX_IMPORT_ERROR}"]

    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        return [], {}, [f"Erreur lors de la lecture DOCX {name}: {exc}"]

    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    chunks = [
        DocumentChunk(text=chunk, meta={"source": name, "type": "docx"})
        for chunk in chunk_text(full_text, max_chars=max_chars, overlap=overlap)
    ]
    summary = {
        "name": name,
        "type": "docx",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, []


def load_file_to_chunks(uploaded_file, max_chars: int = 4000, overlap: int = 400) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    name = uploaded_file.name
    extension = name.split(".")[-1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return [], {}, [f"Format non supporté : {name}"]

    data = uploaded_file.getvalue()

    if extension in TEXT_EXTENSIONS:
        return _ingest_text_document(name, data, extension, max_chars=max_chars, overlap=overlap)
    if extension in CSV_EXTENSIONS:
        return _ingest_csv(name, data, max_chars=max_chars, overlap=overlap)
    if extension in EXCEL_EXTENSIONS:
        return _ingest_excel(name, data, max_chars=max_chars, overlap=overlap)
    if extension in PDF_EXTENSIONS:
        return _ingest_pdf(name, data, max_chars=max_chars, overlap=overlap)
    if extension in DOCX_EXTENSIONS:
        return _ingest_docx(name, data, max_chars=max_chars, overlap=overlap)

    return [], {}, [f"Extension inconnue : {name}"]


def embed_texts(client: OpenAI, model: str, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    response = client.embeddings.create(model=model, input=list(texts))
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return np.ascontiguousarray(normalized.astype(np.float32))


def ensure_faiss_index(dimension: int):
    if faiss is None:
        raise RuntimeError(f"faiss-cpu est requis : {FAISS_IMPORT_ERROR}")
    index = faiss.IndexFlatIP(dimension)
    return index


def add_embeddings_to_index(index, embeddings: np.ndarray):
    if embeddings.size == 0:
        return index
    if index is None:
        index = ensure_faiss_index(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(
    client: OpenAI,
    query: str,
    index,
    texts: Sequence[str],
    metas: Sequence[Dict[str, Any]],
    model: str,
    k: int = 4,
) -> List[Tuple[str, Dict[str, Any], float]]:
    if index is None or not texts:
        return []

    query_embedding = embed_texts(client, model, [query])
    if query_embedding.size == 0:
        return []

    query_array = np.ascontiguousarray(query_embedding)
    distances, indices = index.search(query_array, min(k, len(texts)))
    hits: List[Tuple[str, Dict[str, Any], float]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(texts):
            continue
        hits.append((texts[idx], metas[idx], float(score)))
    return hits


def format_context(hits: Iterable[Tuple[str, Dict[str, Any], float]]) -> str:
    sections: List[str] = []
    for pos, (chunk, meta, _) in enumerate(hits, start=1):
        descriptor_parts = [meta.get("source", "inconnu")]
        if meta.get("page") is not None:
            descriptor_parts.append(f"page {meta['page']}")
        if meta.get("sheet"):
            descriptor_parts.append(f"feuille {meta['sheet']}")
        if meta.get("row_range"):
            descriptor_parts.append(f"lignes {meta['row_range']}")
        descriptor = " ".join(descriptor_parts)
        sections.append(f"[{pos}] Source: {descriptor}\n{chunk}")
    return "\n\n".join(sections)


def format_source_badge(meta: Dict[str, Any], index: int) -> str:
    parts = [f"[{index}] {meta.get('source', 'inconnu')}"]
    if meta.get("page") is not None:
        parts.append(f"page {meta['page']}")
    if meta.get("sheet"):
        parts.append(f"feuille {meta['sheet']}")
    if meta.get("row_range"):
        parts.append(f"lignes {meta['row_range']}")
    return " ".join(parts)


def human_readable_size(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 o"
    units = ["o", "Ko", "Mo", "Go"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"
