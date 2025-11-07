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

## License

This project is released under the [MIT License](LICENSE).
