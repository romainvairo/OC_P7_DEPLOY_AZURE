# OC_P7_DEPLOY_AZURE

Problématique : détecter grâce à des algorithmes de Deep Learning si un tweet a un sentiment positif ou négatif, ce qui pourra servir par la suite à aider les entreprises à atténuer les bad buzz sur les réseaux sociaux

Fichier API du modèle de regression logistique TFIDF avec lemmatisation.

Organisation des fichiers : 
app.py : 
- une fonction clean_text qui permet de nettoyer le texte avant de réaliser une prédiction
- deux points de terminaison :
    un get "/" pour vérifier le fonctionnement de l'API
    un get "/predict" pour obtenir la prédiction en passant une phrase


Deux branches sur ce repository :
- dev avec un yml qui lance 3 tests unitaires qui se trouvent dans le dossier test
- main avec un yml qui envoie les fichiers automatiques dans l'app service azure.

Requirements.txt :
Il est appelé lors de la mise en production dans azure pour installer les packages nécessaires.

log_tfidf_lem.pkl : 
Fichier qui contient notre modèle choisi et qui est utilisé pour faire la prédiction


