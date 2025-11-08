# Tests manuels rapides

## 1. Conversation sans RAG
1. Lancer l’application Streamlit.
2. Envoyer une question simple sans fichiers.
3. Vérifier que la réponse s’affiche et que le rendu/code reste inchangé.

## 2. RAG avec pipeline multi-pass
1. Indexer au moins un document texte.
2. Poser une question liée au document.
3. Confirmer que la réponse suit la structure (Résumé, Détails, Étapes / Code, Sources) et qu’une section "Sources" est présente.
4. Ouvrir l’expander « Diagnostics » pour vérifier les temps de retrieval et des passes.

## 3. Désactivation multi-pass
1. Dans la barre latérale, décocher « Activer multi-pass (2 passes) ».
2. Poser la même question RAG.
3. Vérifier que les diagnostics indiquent "Pass 2 désactivé" et que la réponse reste cohérente.

## 4. Fallback Cross-Encoder
1. Désactiver le reranking Cross-Encoder via la barre latérale.
2. Lancer une requête RAG et vérifier que la réponse est retournée sans erreur.

## 5. Vision / Images
1. Envoyer un message avec une image et vérifier que la pipeline RAG classique continue de fonctionner (pas de régression sur la vision).

## 6. Attachments volumineux
1. Indexer un fichier volumineux (> limite) pour vérifier l’affichage des avertissements et le fallback d’indexation.

Documenter tout comportement inattendu dans les logs Streamlit.
