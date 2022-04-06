#%%
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers,models

IMG_HEIGHT = IMG_WIDTH = 64

###Initialiser avec Glorot uniform comme dans le papier

"""Définition du généréteur de Gaugan avec l'utilisation de spade blocks"""
def spade(segmentation_map,input,filters):
    shape = tf.shape(input)
    resized_map = tf.image.resize(segmentation_map,shape[1:3],method="nearest")
    x = tfa.layers.SpectralNormalization(layers.Conv2D(128,3,strides=1,padding='same',activation='relu'))(resized_map)
    gamma = tfa.layers.SpectralNormalization(layers.Conv2D(filters,3,strides=1,padding='same'))(x)
    beta = tfa.layers.SpectralNormalization(layers.Conv2D(filters,3,strides=1,padding='same'))(x)
    x1 = layers.BatchNormalization()(input)
    x2 = layers.Multiply()([gamma,x1]) + beta
    return x2

def spade_block(segmentation_map,input,filters):
    input_filters = keras.backend.int_shape(input)[-1]
    x = spade(segmentation_map,input,input_filters)
    x = layers.ReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters,3,strides=1,padding='same'))(x)

    x = spade(segmentation_map,x,filters)
    x = layers.ReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters,3,strides=1,padding='same'))(x)
    if filters != input_filters:
        skip = spade(segmentation_map,input,input_filters)
        skip = layers.ReLU()(skip)
        skip = tfa.layers.SpectralNormalization(layers.Conv2D(filters,3,strides=1,padding='same'))(skip)
    else:
        skip = input
    x = layers.Add()([skip,x])
    return x

def define_generator(mask_shape,input_shape=(256,)):
    input = models.Input(shape=input_shape)
    mask_input = models.Input(shape=mask_shape)
    x = layers.Dense(16384)(input)
    x = layers.Reshape((4,4,1024))(x)
    for _ in range(int(IMG_HEIGHT/64)-2):
        x = spade_block(mask_input,x,1024)
        x = layers.UpSampling2D()(x)
    
    x = spade_block(mask_input,x,512)
    x = layers.UpSampling2D()(x)

    x = spade_block(mask_input,x,256)
    x = layers.UpSampling2D()(x)
    x = spade_block(mask_input,x,128)
    x = layers.UpSampling2D()(x)
    x = spade_block(mask_input,x,64)
    x = layers.UpSampling2D()(x)
    
    x = layers.Conv2D(3,3,strides=1,padding='same',activation='tanh')(x)
    model = models.Model([input,mask_input],x)
    return model

#gen = define_generator((256,256,7))
#gen.summary()
#%%
"""Définition du discriminateur de Gaugan type PatchGAN"""
def define_discriminator(input_shape=(64,64,3)):
    inputA = models.Input(shape=input_shape)
    inputB = models.Input(shape=input_shape)
    x = layers.Concatenate()([inputA,inputB])
    x = tfa.layers.SpectralNormalization(layers.Conv2D(64,4,strides=2,padding='same'))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    for i in [128,256]:
        x = tfa.layers.SpectralNormalization(layers.Conv2D(i,4,strides=2,padding='same'))(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(512,4,strides=1,padding='same'))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(1,4,strides=1,padding='same'))(x)
    model = models.Model([inputA,inputB],x)
    return model

#disc = define_discriminator(input_shape=(256,256,3))
#disc.summary()
# %%
"""Définition de l'encodeur pour extraire le style d'une image objectif"""
def define_encoder(input_shape=(64,64,3),latent_dim = 256):
    input = models.Input(shape=input_shape)
    x = layers.Conv2D(64,3,strides=2,padding='same')(input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    for filters in [128,256,512,512,512]:
        x = layers.Conv2D(filters,3,strides=2,padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    mean = layers.Dense(latent_dim,name="mean")(x)
    variance = layers.Dense(latent_dim,name="variance")(x)
    model = models.Model(input,[mean,variance])
    return model