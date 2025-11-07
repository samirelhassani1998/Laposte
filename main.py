"""Streamlit chatbot application powered by OpenAI models."""

from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = "Vous êtes un assistant utile et concis."
DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "o4-mini",
]


def get_api_key() -> str | None:
    """Return the OpenAI API key from Streamlit secrets or environment variables."""
    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    return api_key


def init_session_state() -> None:
    """Initialize default values in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7


def reset_conversation() -> None:
    """Reset the chat history while preserving the system prompt and parameters."""
    st.session_state.messages = []


def render_sidebar() -> None:
    """Render sidebar controls for the chatbot configuration."""
    with st.sidebar:
        st.title("Paramètres")

        st.session_state.system_prompt = st.text_area(
            "Rôle système",
            value=st.session_state.system_prompt,
            help="Définissez le contexte ou les instructions données au modèle.",
            height=120,
        )

        model_index = 0
        if st.session_state.model in AVAILABLE_MODELS:
            model_index = AVAILABLE_MODELS.index(st.session_state.model)

        st.session_state.model = st.selectbox(
            "Modèle OpenAI",
            options=AVAILABLE_MODELS,
            index=model_index,
            help="Choisissez le modèle à utiliser pour la conversation.",
        )

        st.session_state.temperature = st.slider(
            "Température",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.05,
            help="Plus la température est élevée, plus les réponses seront créatives.",
        )

        st.button(
            "Réinitialiser la conversation",
            type="primary",
            use_container_width=True,
            on_click=reset_conversation,
        )

        st.markdown(
            """
            **Astuce :** Définissez votre clé API dans *Streamlit Cloud* via les secrets,
            ou configurez la variable d'environnement `OPENAI_API_KEY` en local.
            """
        )


def build_prompt_history(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Construct the message payload to send to OpenAI."""
    messages: List[Dict[str, str]] = []
    system_prompt = st.session_state.system_prompt.strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(conversation)
    return messages


def generate_response(client: OpenAI, conversation: List[Dict[str, str]]) -> str:
    """Call the OpenAI API and return the assistant response."""
    response = client.chat.completions.create(
        model=st.session_state.model,
        temperature=st.session_state.temperature,
        messages=build_prompt_history(conversation),
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    st.set_page_config(page_title="Chatbot OpenAI", page_icon="💬", layout="wide")
    st.title("🤖 Chatbot OpenAI")
    st.write("Discutez avec un modèle OpenAI directement depuis cette application Streamlit.")

    init_session_state()
    render_sidebar()

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "⚠️ Aucune clé API OpenAI détectée. Ajoutez `OPENAI_API_KEY` dans vos secrets ou variables d'environnement pour utiliser l'application."
        )
        st.stop()

    client = OpenAI(api_key=api_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Entrez votre message…")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        conversation = st.session_state.messages + [{"role": "user", "content": user_input}]

        try:
            assistant_reply = generate_response(client, conversation)
        except Exception as exc:
            with st.chat_message("assistant"):
                st.error(f"Une erreur est survenue lors de l'appel à l'API : {exc}")
            return

        conversation.append({"role": "assistant", "content": assistant_reply})
        st.session_state.messages = conversation

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)


if __name__ == "__main__":
    main()
