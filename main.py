import os
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

st.set_page_config(page_title="ChatGPT-like Chatbot", layout="wide")

AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-5",
]


def _init_session_state() -> None:
    if "api_key" not in st.session_state:
        env_key = os.getenv("OPENAI_API_KEY")
        st.session_state.api_key = env_key if env_key else None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]


def _reset_chat() -> None:
    st.session_state.messages = []


def _remove_api_key() -> None:
    st.session_state.api_key = None
    _reset_chat()
    st.rerun()


def _render_key_gate() -> None:
    st.markdown(
        """
        <style>
            .login-wrapper {
                max-width: 420px;
                margin: 10vh auto;
                padding: 2.5rem;
                background: #ffffff;
                border-radius: 18px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
            }
            .login-wrapper h1 {
                text-align: center;
                margin-bottom: 0.5rem;
            }
            .login-wrapper p {
                text-align: center;
                color: #6b7280;
                margin-bottom: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.form("api-key-form", clear_on_submit=False):
        st.markdown(
            """
            <div class='login-wrapper'>
                <h1>Bienvenue</h1>
                <p>Entrez votre clé API OpenAI pour démarrer la conversation.</p>
            """,
            unsafe_allow_html=True,
        )
        api_key_input = st.text_input(
            "Clé API OpenAI",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )
        submit = st.form_submit_button("Continuer", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if not api_key_input:
            st.error("Merci de renseigner une clé API valide.")
        else:
            st.session_state.api_key = api_key_input.strip()
            _reset_chat()
            st.rerun()



def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Paramètres")
        try:
            default_index = AVAILABLE_MODELS.index(st.session_state.selected_model)
        except ValueError:
            default_index = 0
        model = st.selectbox("Modèle", AVAILABLE_MODELS, index=default_index)
        st.session_state.selected_model = model

        st.button("Nouvelle conversation", on_click=_reset_chat)
        st.button("Changer de clé API", on_click=_remove_api_key)

        st.markdown("---")
        st.caption("Votre clé n'est jamais sauvegardée côté serveur.")


def _call_openai(messages: List[Dict[str, str]]) -> str:
    client = OpenAI(api_key=st.session_state.api_key)
    response = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _render_chat_interface() -> None:
    _render_sidebar()

    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem !important; padding-bottom: 6rem !important;}
        .stChatFloatingInputContainer {bottom: 1.5rem;}
        div[data-testid="stChatMessage"] {background: transparent;}
        div[data-testid="stChatMessageUser"] > div:nth-child(1) {
            background: #e7f5f0;
            color: #0f172a;
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stChatMessageAssistant"] > div:nth-child(1) {
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        .chat-header {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        .empty-state {
            margin-top: 15vh;
            text-align: center;
            color: #6b7280;
        }
        .empty-state h2 {
            color: #111827;
            font-size: 2.25rem;
            margin-bottom: 0.5rem;
        }
        .empty-state p {
            margin: 0.25rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='chat-header'>ChatGPT-like Chatbot</div>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages:
        st.markdown(
            """
            <div class='empty-state'>
                <h2>Que voulez-vous savoir aujourd'hui ?</h2>
                <p>Choisissez un modèle dans la barre latérale et lancez la discussion.</p>
                <p>Votre historique reste visible dans cette fenêtre, comme sur ChatGPT.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    prompt = st.chat_input("Envoyer un message")
    if prompt and prompt.strip():
        user_message = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                reply = _call_openai(st.session_state.messages)
            except Exception as error:  # noqa: BLE001 - handled gracefully for the UI
                placeholder.error(f"Erreur lors de l'appel à l'API : {error}")
            else:
                placeholder.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    _init_session_state()

    if not st.session_state.api_key:
        _render_key_gate()
    else:
        _render_chat_interface()
