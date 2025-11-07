# ChatGPT-like Streamlit Chatbot

This project recreates a minimal ChatGPT-style user interface using Streamlit and the OpenAI API. Users can supply their own API key directly in the app, pick a model, and begin chatting immediately.

## Features

- **Inline API key capture**: Provide your OpenAI API key inside the app—no need for `st.secrets`.
- **Model selection**: Switch between available models such as GPT-4o, GPT-4o mini, and GPT-5 (if enabled on your account).
- **Clean conversation view**: Messages are displayed in a vertically stacked chat log with a chat input area at the bottom of the screen.
- **Session persistence**: The API key and conversation history live in `st.session_state` during the browsing session.

## Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key with access to the desired models

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run main.py
```

Open the provided local URL in your browser. The app will first ask for your API key. Once provided, it reveals the chat interface.

### Environment Variables (Optional)

If you do not want to type your API key every time, you can create a `.env` file at the project root and set `OPENAI_API_KEY=...`. The app will load it automatically on startup, but the key can still be changed from the UI at any time.

## Customization

- Update the list of models in `main.py` to match the ones available on your account.
- Adjust the theme colors in `.streamlit/config.toml` to tweak the look and feel.

## RAG & Upload

The chat interface now includes an optional retrieval-augmented generation (RAG) workflow:

- **Drag & drop documents** directly in the sidebar (`csv`, `xlsx`, `xls`, `pdf`, `docx`, `txt`, `md`). Up to five files at a time and 20&nbsp;MB per file are accepted.
- **On-demand indexing** builds an in-memory FAISS index for the current session using OpenAI's `text-embedding-3-large` model. Files are read in memory only; nothing is persisted on disk and the API key never leaves the session.
- **Chunking & metadata**: each document is normalized, chunked (~4 000 chars with 400-char overlap), and enriched with metadata (source file, page/sheet/row range when applicable).
- **Contextual answers**: when the index is populated, every new user question retrieves the top-4 chunks and injects them into the system prompt. Responses cite their sources and a badge indicates when RAG is active.
- **Reset anytime**: use the “Réinitialiser base” button to clear the FAISS index and associated documents from the session state.

The sidebar summarises the indexed corpus (file sizes, estimated tokens, chunk counts, embedding model). If a PDF contains no extractable text (e.g. scanned documents), the app warns you and skips it.

## License

This project is released under the [MIT License](LICENSE).
