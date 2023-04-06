# Projet_IA_groupe_ANC : Développer un site web Django exploitant de l'IA


![image](https://user-images.githubusercontent.com/59603301/230472965-7796bd38-3adf-4524-8f4c-d0ae163857de.png)



### Contexte du projet

IAOuverte est une petite startup qui veut rendre l'IA accessible à tous. Son fondateur, Samuel AltHomme, a pensé à un outil permettant à des utilisateurs d'uploader des données dans différents formats (csv, excel...), de choisir une tâche (régression, classification, clustering), un ou plusieurs algorithmes de la bibliothèque scikit learn et d'obtenir un résultat. Des graphiques viendront améliorer l'exploitation de ces résultats. Il compte nommer cet outil IA4All. 

Ce site web contient alors des explications simples sur le fonctionnement du machine learning et des modèles disponibles. La prédiction n'est accéssible qu'aux utilisateurs inscrits  mais les pages d'informations seront accessibles à toute personne naviguant sur le site. Un véritable interface utilisateur est présentée, il est possible de réinitialiser son mot de passe grâce à un envoie d'email. De plus, une sauvegarde de toutes ses prédictions est disponoble ce qui permet de garder un historique.

Les administrateurs du site ont tous les droits (ajout, suppression du compte utilisateur).

Ce projet présente :
    - Un site web Django
    - 3 modèles par tâches (régression, classification, clustering)
    - Une interface Accueil Utilisateur (inscription, la connexion, la déconnexion et la suppression de son compte sont possibles)
    - Une interface application de prédiction interactive ( boutons, choix de la target, affichage des métriques, et graphiques)



### Utilisaton

- Cloner ce projet 
https://github.com/Cesarp27/Projet_IA_groupe_ANC.git


- Démarrer le site web
Pour démarrer le serveur local, il suffit d'exécuter la commande ci-dessous dans le répertoire du projet (dossier où se trouver manage.py) :

python3 manage.py runserver

Dès que le serveur est actif, vous pouvez utiliser votre navigateur est accéder à l'URL http://127.0.0.1:8000/.



En espérant que vous apprécierez notre site Django.
Inscrivez vous ! 


Projet réalisé par :

    - Célia Mato
    - Noli Sapunxhiu
    - César Parra
