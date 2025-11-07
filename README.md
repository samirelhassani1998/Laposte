# Chatbot Streamlit + OpenAI

Application web construite avec [Streamlit](https://streamlit.io/) permettant de discuter avec les modèles OpenAI les plus récents. L'interface offre un historique de conversation complet, un rôle système éditable et des paramètres ajustables pour le modèle et la température.

## ✨ Fonctionnalités
- Interface conversationnelle moderne avec `st.chat_message` et `st.chat_input`.
- Rôle système personnalisable et sauvegarde de l'historique dans la session Streamlit.
- Choix du modèle OpenAI et réglage de la température.
- Bouton de réinitialisation pour repartir d'une conversation vierge.
- Gestion sécurisée de la clé API via `st.secrets` ou la variable d'environnement `OPENAI_API_KEY`.

## 🚀 Démarrage rapide

### Prérequis
- Python 3.11 (recommandé)
- Une clé API OpenAI valide.

### Installation locale
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\\Scripts\\activate sur Windows
pip install -r requirements.txt
```

Créez ensuite un fichier `.streamlit/secrets.toml` en vous inspirant de `.streamlit/secrets.example.toml` :
```toml
OPENAI_API_KEY = "votre_cle_api"
```

Lancez l'application :
```bash
streamlit run main.py
```

## ☁️ Déploiement sur Streamlit Cloud
1. Poussez ce dépôt vers GitHub.
2. Dans Streamlit Cloud, créez une nouvelle app pointant vers `main.py`.
3. Ajoutez la clé API dans la section **Secrets** du projet (`OPENAI_API_KEY`).
4. Déployez : l'application est prête !

## 📁 Structure du projet
```
.
├── main.py
├── pages/
│   └── 01_About.md
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.example.toml
└── runtime.txt
```

## 🛡️ Licence
Ce projet est distribué sous licence [MIT](LICENSE).
