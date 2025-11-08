SYSTEM_BASE = """Tu es un assistant expert et pédagogue.
Réponds de façon structurée, précise et complète. Utilise le contexte fourni (RAG) et indique les limites si le contexte est insuffisant.
Structure attends:
### Résumé
### Détails
### Étapes / Code (avec des blocs ```lang)
### Sources (titres + identifiants)"""

USER_TEMPLATE = """Contexte RAG (top {k} passages):
{context}

Historique condensé:
{history}

Question:
{query}

Consignes:
- Ne pas inventer de sources. Cite les passages pertinents sous 'Sources'.
- Si un code est nécessaire, retourne un seul bloc ```lang complet.
"""

IMPROVE_TEMPLATE = """Améliore et restructure la réponse ci-dessous:
- Plus claire, plus complète, sections conformes, code en ```lang si présent.
- Ajoute 'Limites / Doutes' si utile.
Réponse initiale:
{draft}
"""
