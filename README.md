# Waifu Generator

Projet de génération de personnages d'anime à l'aide de GAN principalement.

Une première étape est d'utiliser un DCGAN pour générer des visages de manière aléatoire.

Ensuite l'objectif est de générer des personnages réalistes à partir de masques de segmentation pouvant être dessinés à la main. Ceci se fait avec GauGAN.

## DCGAN

- DCGAN pour générer des waifus en 64 sur 64 et autres tests infructueux (PROGAN,ResNetGAN)
- baseline.py : Notebook à compléter pour la première phase du projet
- Résultats obtenus :
![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/generated_images_e064.png?raw=true)

## Segmentation de waifus

- GauGAN nécessite un dataset de paires (masques de segmentation|images réelles)
pour être entraîné. Or il n'existe pas de dataset public disponible donc
nous avons labellisé quelques centaines d'images nous-même. Ceci étant
encore insuffisant pour entraîner un GAN, nous avons entraîné un modèle de
segmentation (Unet puis DeepLabV3Plus) pour générer des masques de segmentation
pour le dataset https://www.kaggle.com/datasets/lukexng/animefaces-512x512 
constitué de 140 000 images en résolution 512*512. 

- Tutoriels utiles:
    - U-net : https://keras.io/examples/vision/oxford_pets_image_segmentation/
    - Transfer learning mobilenetv2 : https://www.tensorflow.org/tutorials/images/segmentation
    ![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/u-net-architecture-1024x682.png)
    - DeepLabV3plus: https://keras.io/examples/vision/deeplabv3_plus/
    ![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/1%202mYfKnsX1IqCCSItxpXSGA.png)

- Fichiers python disponibles:
    - deeplabv3.py : modèle DeepLabV3plus basé sur des convolutions à trous
    et du Spatial Pyramidal Pooling avec comme backbone MobileNetV2 +
    subclass model keras
    - unet.py : modèle U-Net avec comme backbone pour l'encodeur MobileNetV2
    + subclass model
    - losses.py : loss à utiliser parmi lesquelles Dice, Dice_BCE, Dice_CE_focal
    - segmentation_pipeline.py: utilitaire pour créer le dataset de segmentation
    sous la forme image + masque one-hot encoded + masque RGB
    - main_seg.py : fichier principal pour entraîner le modèle choisi et le tester

- Comment sauvegarder les images annotées:
    - Mettre les images d'origine dans "segmentation_waifus/images/training/"
    - Mettre les masques de segmentation dans "segmentation-waifus/annotations/training/"

- Masques obtenus: 

![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/output.png)
## Génération de waifus HD conditionnée

- Utilisation de GauGAN basé sur la Spatial Adaptative Normalization (SPADE) 
pour générer des images à partir de masques de segmentation.
Nous avons comme dans le papier ajouté aussi un encodeur pour inverser
le GAN en même temps et pouvoir utiliser des images de référence afin
de conditionner la génération des personnages.

- Tuto architecture GauGAN : https://keras.io/examples/generative/gaugan/
![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/spade_layer.png)
![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/D-YnPm-WwAAvG_G.jpg)

- Fichiers python disponibles:
    - models.py : architecture des modèles utilisés (encodeur, générateur et
    discriminateur)
    - losses_gaugan.py : loss utilisées pour entraîner le modèle parmi lesquelles
    une loss pixel par pixel, la KL divergence, une loss de contenu, une loss
    de contenu avec des features extraites par VGG19 et la hinge loss pour le 
    discriminateur
    - gaugan_class.py : subclass model GauGAN assez complexe 
    - segmentation_pipeline_bis.py : réplique de la pipeline de segmentation
    adapté au format pour GauGAN
    - Gaugan.py : fichier principal pour entraîner le modèle et le tester

- Objectif:

![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/gaugan_steins.png)
## GUI de qualité

*A faire*
- Programme pour dessiner sur une image avec des couleurs spécifiques (celles de la segmentation)
- Importer un masque de segmentation déjà fait à modifier avec le programme précédent
- (Choix de couleurs pour une partie)
- Coloriage d'une partie contourée (genre bouton remplissage de forme)
- Prédiction du modèle sur le masque après tout ce traitement
- Affichage de l'image générée à droite (éventuellement en direct si possible)

![alt text](https://github.com/Rubiksman78/Waifu_generator/blob/main/images/guisteins.jpg)

## Papiers

Ensemble d'articles de recherche utilisés pour ce projet.
