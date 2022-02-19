# Waifu_generator

Projet de génération de personnages d'anime à l'aide de GAN principalement.

Une première étape est d'utiliser un DCGAN pour générer des visages de manière aléatoire.

Ensuite l'objectif est de générer des personnages réalistes à partir de masques de segmentation pouvant être dessinés à la main. Ceci se fait avec GauGAN.

## DCGAN

- DCGAN pour générer des waifus en 64 sur 64 et autres tests infructueux (PROGAN,ResNetGAN)
- Utilitaire pour extraire un dataset d'un dossier d'images

## Segmentation de waifus

Segementation avec U-NET. 

Améliorations : DICE loss, autres architecturess
- Labels et leur couleur RGB associée
- Waifus et leurs masques de segmentation en couleur 
- Utilitaire pour créer un dataset avec image réelle + masque de segmentation + masque one - hot
- 
- Comment sauvegarder les images annotées:
    - Mettre les images d'origine dans segmentation_waifus/images/training/
    - Mettre les masques de segmentation dans segmentation-waifus/annotations/training/
   
## Génération de waifus HD conditionnée

- GauGAN

## GUI de qualité

