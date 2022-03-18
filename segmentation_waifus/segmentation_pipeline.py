import tensorflow as tf
import numpy as np
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

### Grosse Data augment ###
class Augment(tf.keras.layers.Layer):
    #Data Augment : rotation + flip 
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.25,fill_mode = "constant",seed=seed)])
        self.augment_labels = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.25,fill_mode = "constant",seed=seed)])
        self.augment_true_labels = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.25,fill_mode = "constant",seed=seed)])    
    
    def apply_random_brightness(self, image, mask,true_mask):
        condition = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(
            condition, lambda: tf.image.random_brightness(
            image,0.4),
            lambda: tf.identity(image)
        )
        return image, mask,true_mask
    
    def apply_random_contrast(self, image, mask,true_mask):
        condition = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(
            condition, lambda: tf.image.random_contrast(
                image,0.75,
                1.5
            ), lambda: tf.identity(image)
        )
        return image, mask,true_mask
    
    def apply_random_saturation(self, image, mask,true_mask):
        condition = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(
            condition, lambda: tf.image.random_saturation(
            image, 0.75,
                1.25
            ), lambda: tf.identity(image)
        )
        return image, mask,true_mask

    def apply_noise(self,image,mask,true_mask):
        condition = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        noise = tf.random.normal(shape=tf.shape(image),stddev=0.075)
        image = tf.cond(
            condition, lambda: tf.add(image,noise),
            lambda: tf.identity(image)
        )
        return image, mask,true_mask

    def apply_hue(self,image,mask,true_mask):
        condition = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(
            condition, lambda: tf.image.random_hue(
            image, 0.1
            ), lambda: tf.identity(image)
        )
        return image, mask,true_mask

    def apply_resize_crop(self,image,mask,true_mask):
        NUM_BOXES = tf.shape(image)[0]
        CROP_SIZE = (256,256)
        condition = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
        box_indices = tf.range(0, NUM_BOXES, dtype=tf.int32)
        image = tf.cond(
            condition, lambda: tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE),
            lambda: tf.identity(image)
        )
        mask =  tf.cond(
            condition, lambda: tf.image.crop_and_resize(mask, boxes, box_indices, CROP_SIZE),
            lambda: tf.identity(mask)
        )
        true_mask =  tf.cond(
            condition, lambda: tf.image.crop_and_resize(true_mask, boxes, box_indices, CROP_SIZE),
            lambda: tf.identity(true_mask)
        )
        return image,mask,true_mask

    def crop_size(self,image,mask,true_mask):
        size = np.random.randint(128,164)
        batch_size = tf.shape(image)[0]
        size = [batch_size,size,size,3]
        condition = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        im_size = tf.shape(image)[1:3]
        image = tf.cond(
            condition, lambda: tf.image.random_crop(image,size,seed=42),
            lambda: tf.identity(image)
        )
        mask =  tf.cond(
            condition, lambda: tf.image.random_crop(mask,size,seed=42),
            lambda: tf.identity(mask)
        )
        true_mask =  tf.cond(
            condition, lambda: tf.image.random_crop(true_mask,size,seed=42),
            lambda: tf.identity(true_mask)
        )
        image = tf.image.resize(image,size=im_size)
        mask = tf.image.resize(mask,size=im_size)
        true_mask = tf.image.resize(true_mask,size=im_size)
        return image,mask,true_mask

    def call(self, inputs, labels ,true_labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        true_labels = self.augment_true_labels(true_labels)
        inputs,labels,true_labels = self.apply_random_brightness(inputs,labels,true_labels)
        inputs,labels,true_labels = self.apply_random_contrast(inputs,labels,true_labels)
        inputs,labels,true_labels = self.apply_random_saturation(inputs,labels,true_labels)
        inputs,labels,true_labels = self.apply_hue(inputs,labels,true_labels)
        inputs,labels,true_labels = self.apply_noise(inputs,labels,true_labels)
        #inputs,labels,true_labels = self.apply_resize_crop(inputs,labels,true_labels)
        inputs,labels,true_labels = self.crop_size(inputs,labels,true_labels)
        return inputs, labels, true_labels

class One_Hot(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs, labels ,true_labels):
        res = normalize_maskbis(labels)
        return inputs, res, true_labels
