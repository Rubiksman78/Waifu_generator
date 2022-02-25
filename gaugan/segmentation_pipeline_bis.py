import tensorflow as tf
import numpy as np
import cv2
import glob
from tensorflow import keras
from keras import layers

hair = str([66, 64, 183])
eyes = str([38,153,174])
clothes = str([128,32,0])
face = str([144,232,206])
skin = str([99,169,77])
background = str([0,0,0])
mouth = str([255,232,177])
label = {background:0, hair:1, eyes:2, clothes:3,face:4,skin:5,mouth:6}

palette = np.array([
    [66, 64, 183],
    [38,153,174],
    [128,32,0],
    [144,232,206],
    [99,169,77],
    [0,0,0],
    [255,232,177]
],np.uint8)

def inv_mask(mask):#Prend un masque one-hot encoded et renvoie son équivalent image couleur
    global palette
    size = mask.shape[1]
    palette2 = tf.constant(palette,dtype=tf.uint8)
    class_indexes = tf.argmax(mask, axis=-1)
    # NOTE this operation flattens class_indexes
    class_indexes = tf.reshape(class_indexes, [-1])
    color_image = tf.gather(palette2, class_indexes)
    color_image = tf.reshape(color_image, [size, size, 3])
    return color_image

def normalize_maskbis(img):#Prend un masque couleur et renvoie son équivalent avec seulement les classes voulues
    #(W,H,255) -> (W,H,num_classes)
    semantic_map = []
    for colour in palette:
        class_map = tf.reduce_all(tf.equal(img, colour), axis=-1)
        semantic_map.append(class_map)
    semantic_map = tf.stack(semantic_map, axis=-1)
    semantic_map = tf.cast(semantic_map, tf.float32) 
    return semantic_map 

def parse_image(img_path: str) -> dict:
    #Renvoie un dictionnaire image + masque pour être encodé + masque couleur pour l'affichage
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    # For one Image path:
    # .../trainset/images/training/image1.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/image2.png
    
    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)

    label_path = tf.strings.regex_replace(mask_path, "images", "annotations")
    label_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    label_mask = tf.io.read_file(label_path)
    label_mask = tf.image.decode_png(label_mask, channels=3)
    return {'image': image, 'segmentation_mask': label_mask,"true_mask":mask}

def normalize(input_image):
    #Normalise [0,255] -> [0,1]
    input_image = tf.cast(input_image,tf.float32)/255.0
    return input_image

def load_image(datapoint,im_size=512):
    #Redimensionne toutes les images et normalise l'image d'origine
    input_image = tf.image.resize(datapoint['image'],(im_size,im_size))
    input_mask = tf.image.resize(datapoint['segmentation_mask'],(im_size,im_size))
    true_mask = tf.image.resize(datapoint['true_mask'],(im_size,im_size))
    input_image = normalize(input_image)
    return input_image,input_mask,true_mask
class Augment(tf.keras.layers.Layer):
    #Data Augment : rotation + flip 
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.1,fill_mode = "constant",seed=seed)])
        self.augment_labels = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.1,fill_mode = "constant",seed=seed)])
        self.augment_true_labels = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.1,fill_mode = "constant",seed=seed)])    
    
    def call(self, inputs, labels ,true_labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        true_labels = self.augment_true_labels(true_labels)
        return inputs, labels, true_labels

class One_Hot(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs, labels ,true_labels):
        res = normalize_maskbis(labels)
        return inputs, res, true_labels

class One_Hot_bis(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs, labels ,true_labels):
        res = normalize_maskbis(labels)
        return inputs, true_labels, res