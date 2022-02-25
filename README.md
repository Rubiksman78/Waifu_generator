# Waifu_generator

Projet de génération de personnages d'anime à l'aide de GAN principalement.

Une première étape est d'utiliser un DCGAN pour générer des visages de manière aléatoire.

Ensuite l'objectif est de générer des personnages réalistes à partir de masques de segmentation pouvant être dessinés à la main. Ceci se fait avec GauGAN.

## DCGAN

- DCGAN pour générer des waifus en 64 sur 64 et autres tests infructueux (PROGAN,ResNetGAN)
- Utilitaire pour extraire un dataset d'un dossier d'images
- Résultats obtenus :

![alt text](https://github.com/Rubiksman78/Waifu_generator/tree/main/images/generated_images_e064.png?raw=true)

## Segmentation de waifus

Segementation avec U-NET. 

Tuto U-net : https://keras.io/examples/vision/oxford_pets_image_segmentation/

Tuto transfer learning mobilenetv2 : https://www.tensorflow.org/tutorials/images/segmentation

Améliorations : DICE loss, autres architecturess (Deeplab, convolutions atrous)
- Labels et leur couleur RGB associée
- Waifus et leurs masques de segmentation en couleur 
- Utilitaire pour créer un dataset avec image réelle + masque de segmentation + masque one - hot
- Diverses autres fonctions (loss,model,test_seg)
- Comment sauvegarder les images annotées:
    - Mettre les images d'origine dans "segmentation_waifus/images/training/"
    - Mettre les masques de segmentation dans "segmentation-waifus/annotations/training/"
   
## Génération de waifus HD conditionnée

- GauGAN
- Tuto architecture de base : https://keras.io/examples/generative/gaugan/
## GUI de qualité

- Programme pour dessiner sur une image avec des couleurs spécifiques (celles de la segmentation)
- Importer un masque de segmentation déjà fait à modifier avec le programme précédent
- (Choix de couleurs pour une partie)
- Coloriage d'une partie contourée (genre bouton remplissage de forme)
- Prédiction du modèle sur le masque après tout ce traitement
- Affichage de l'image générée à droite (éventuellement en direct si possible)