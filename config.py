from dataclasses import dataclass


@dataclass(frozen=True)
class PerfConfig:
    # Modèle
    default_model: str = "gpt-4o-mini"

    # RAG
    rag_k: int = 4
    use_mmr: bool = True
    mmr_fetch_k: int = 24
    mmr_lambda: float = 0.5
    use_reranker: bool = False
    use_multipass: bool = False

    # Génération
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 900
    streaming: bool = True

    # Quality escalation (désactivée par défaut)
    quality_escalation: bool = False
