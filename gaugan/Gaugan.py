#%%
import tensorflow as tf
from tensorflow import keras
from keras import layers,models,losses
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import cv2
from glob import glob
from PIL import Image
from xgboost import train
from segmentation_pipeline_bis import * 
#%%
perso_path = 'C:/SAMUEL/Centrale/Automatants/Waifu_generator/' #Mettre votre path local vers le repo
dataset_path = perso_path + 'segmentation_waifus/images/'

BATCH_SIZE = 4
IMG_HEIGHT = IMG_WIDTH = 128
NUM_CLASSES = 7
buffer_size = 100

def define_dataset(dataset_path, batch_size, buffer_size):
    training_data = "training/"
    val_data = "validation/"
    TRAINSET_SIZE = len(glob.glob(dataset_path + training_data + "*.jpg"))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    VALSET_SIZE = len(glob.glob(dataset_path + val_data + "*.jpg"))
    print(f"The Validation Dataset contains {VALSET_SIZE} images.")

    train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + "*.jpg")
    train_dataset = train_dataset.map(parse_image)

    val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg")
    val_dataset = val_dataset.map(parse_image)
    dataset = {"train": train_dataset, "val": val_dataset}

    train_images = dataset['train'].map(
        lambda x: load_image(x,im_size=IMG_HEIGHT), num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['val'].map(
        lambda x: load_image(x,im_size=IMG_HEIGHT), num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_images
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size,drop_remainder=True)
        .map(Augment())
        .map(One_Hot_bis())
        .repeat(20)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(batch_size,drop_remainder=True).map(One_Hot_bis())
    return train_batches, test_batches

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

train_batches,test_batches = define_dataset(dataset_path,BATCH_SIZE,buffer_size)
print(train_batches)
for images,true_masks,masks in train_batches.take(3):
    sample_image,sample_mask = images[0],true_masks[0]
    display([sample_image,sample_mask])

sample_train_batch = next(iter(train_batches))
print(f"Segmentation map batch shape: {sample_train_batch[0].shape}.")
print(f"Image batch shape: {sample_train_batch[1].shape}.")
print(f"One-hot encoded label map shape: {sample_train_batch[2].shape}.")

# Plot a view samples from the training set.
for segmentation_map, real_image in zip(sample_train_batch[1], sample_train_batch[0]):
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1).set_title("Segmentation Map")
    plt.imshow(segmentation_map/255)
    fig.add_subplot(1, 2, 2).set_title("Real Image")
    plt.imshow((real_image + 1) / 2)
    plt.show()
#%%
class GaussianSampler(layers.Layer):
    def __init__(self,batch_size,latent_dim,**kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size 
        self.latent_dim = latent_dim

    def call(self,inputs):
        means,variance = inputs
        epsilon = tf.random.normal(shape=(self.batch_size,self.latent_dim),mean=0.0,stddev = 1.0)
        samples = means + tf.exp(0.5 * variance) * epsilon
        return samples
#%%
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
    
    for _ in range(1):
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

gen = define_generator((512,512,10))
gen.summary()
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
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(1,4,strides=1,padding='same'))(x)
    model = models.Model([inputA,inputB],x)
    return model

disc = define_discriminator()
disc.summary()
# %%
"""Définition de l'encodeur pour extraire le style d'une image objectif"""
def define_encoder(input_shape=(64,64,3),latent_dim = 256):
    input = models.Input(shape=input_shape)
    x = layers.Conv2D(64,3,strides=2,padding='same')(input)
    x = tfa.layers.InstanceNormalization()(x)
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
    
encoder = define_encoder()
encoder.summary()

"""Définition des différentes loss utilisées"""
def gen_loss(y):
    return -tf.reduce_mean(y)

def kl_divergence_loss(mean,variance):
    return -0.5*tf.reduce_sum(1+variance-tf.square(mean)-tf.exp(variance))

class FeatureMatchingLoss(keras.losses.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.mae = losses.MeanAbsoluteError()

    def call(self,y_true,y_pred):
        loss = 0
        for i in range(y_true.shape[0]-1):
            loss += self.mae(y_true[i],y_pred[i])
        return loss

class VGGFeatureMatchingLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = keras.Model(vgg.input, layer_outputs, name="VGG")
        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        y_true = keras.applications.vgg19.preprocess_input(127.5 * (y_true + 1))
        y_pred = keras.applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
        real_features = self.vgg_model(y_true)
        fake_features = self.vgg_model(y_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
        return loss

class DiscriminatorLoss(keras.losses.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.hinge_loss = keras.losses.Hinge()

    def call(self,y,is_real):
        label = 1.0 if is_real else -1.0
        return self.hinge_loss(label,y)

class GauGAN(keras.models.Model):
    def __init__(
        self,
        image_size,
        num_classes,
        batch_size,
        latent_dim,
        feature_loss_coeff = 10,
        vgg_feature_loss_coeff = 0.1,
        kl_divergence_loss_coeff = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.feature_loss_coeff = feature_loss_coeff
        self.vgg_feature_loss_coeff = vgg_feature_loss_coeff
        self.kl_divergence_coeff = kl_divergence_loss_coeff
        self.image_shape = (image_size,image_size,3)
        self.mask_shape = (image_size,image_size,num_classes)

        self.discriminator = define_discriminator(input_shape=self.image_shape)
        self.generator = define_generator(self.mask_shape,input_shape=(latent_dim,))
        self.encoder = define_encoder(input_shape = self.image_shape,latent_dim=latent_dim)
        self.sampler = GaussianSampler(batch_size,latent_dim)
        self.patch_size,self.combined_model = self.build_combined_generator()

        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
        self.feat_loss_tracker = tf.keras.metrics.Mean(name="feat_loss")
        self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.gen_loss_tracker,
            self.feat_loss_tracker,
            self.vgg_loss_tracker,
            self.kl_loss_tracker,
        ]

    def build_combined_generator(self):
        self.discriminator.trainable = False
        mask_input = models.Input(shape=self.mask_shape,name="mask")
        image_input = models.Input(shape=self.image_shape,name="image")
        latent_input = models.Input(shape=(self.latent_dim,),name="latent")
        generated_image = self.generator([latent_input,mask_input],training=True)
        discriminator_output = self.discriminator([image_input,generated_image],training=False)
        patch_size = discriminator_output[-1].shape[1]
        combined_model = models.Model(
            [latent_input,mask_input,image_input],
            [discriminator_output,generated_image]
        )
        return patch_size, combined_model

    def compile(self,gen_lr=4e-4,disc_lr=4e-4,**kwargs):
        super().compile(**kwargs)
        self.generator_optimizer = keras.optimizers.Adam(gen_lr,beta_1 = 0.0)
        self.discriminator_optimizer = keras.optimizers.Adam(disc_lr,beta_1=0.0)
        self.discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()

    def train_discriminator(self,latent_vector,segmentation_map,real_image,labels):
        fake_images = self.generator([latent_vector,labels])
        with tf.GradientTape() as gradient_tape:
            pred_fake = self.discriminator([segmentation_map,fake_images],training=True)[-1]
            pred_real = self.discriminator([segmentation_map,real_image],training=True)[-1]
            loss_fake = self.discriminator_loss(pred_fake,False)
            loss_real = self.discriminator_loss(pred_real,True)
            total_loss = 0.5 * (loss_fake+loss_real)
        self.discriminator.trainable = True
        gradients = gradient_tape.gradient(total_loss,self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients,self.discriminator.trainable_variables))
        return total_loss

    def train_generator(self,latent_vector,segmentation_map,labels,image,mean,variance):
        self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            real_output = self.discriminator([segmentation_map,image],training=False)
            fake_output,fake_image = self.combined_model([latent_vector,labels,segmentation_map])
            pred = fake_output[-1]
            g_loss = gen_loss(pred)
            kl_loss = self.kl_divergence_coeff * kl_divergence_loss(mean,variance)
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image,fake_image)
            feature_loss = self.feature_loss_coeff * self.feature_matching_loss(real_output,fake_output)
            total_loss = g_loss + kl_loss + vgg_loss + feature_loss
        gradients = tape.gradient(total_loss,self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients,self.generator.trainable_variables))
        return total_loss,feature_loss,vgg_loss,kl_loss

    def train_step(self,data):
        image,segmentation_map,labels=data
        mean,variance = self.encoder(image,training=True)
        latent_vector = self.sampler([mean,variance])
        discriminator_loss = self.train_discriminator(latent_vector,segmentation_map,image,labels)
        (generator_loss,feature_loss,vgg_loss,kl_loss) = self.train_generator(latent_vector,segmentation_map,labels,image,mean,variance)

        self.disc_loss_tracker.update_state(discriminator_loss)
        self.gen_loss_tracker.update_state(generator_loss)
        self.feat_loss_tracker.update_state(feature_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        results = {m.name : m.result() for m in self.metrics}
        return results

    def test_step(self,data):
        segmentation_map,image,labels=data
        mean,variance = self.encoder(image)
        latent_vector = self.sampler([mean,variance])
        fake_images = self.generator([latent_vector,labels])
        pred_fake = self.discriminator([segmentation_map,fake_images])[-1]
        pred_real = self.discriminator([segmentation_map,image])[-1]
        loss_fake = self.discriminator_loss(pred_fake,False)
        loss_real = self.discriminator_loss(pred_real,True)
        total_discriminator_loss = 0.5 * (loss_fake+loss_real)
        
        real_output = self.discriminator([segmentation_map,image])
        fake_output,fake_image = self.combined_model([latent_vector,labels,segmentation_map])
        pred = fake_output[-1]
        g_loss = gen_loss(pred)
        kl_loss = self.kl_divergence_coeff * kl_divergence_loss(mean,variance)
        vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image,fake_image)
        feature_loss = self.feature_loss_coeff * self.feature_matching_loss(real_output,fake_output)
        total_loss = g_loss + kl_loss + vgg_loss + feature_loss

        self.disc_loss_tracker.update_state(total_discriminator_loss)
        self.gen_loss_tracker.update_state(total_loss)
        self.feat_loss_tracker.update_state(feature_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        results = {m.name : m.result() for m in self.metrics}
        return results
    
    def call(self,inputs):
        latent_vectors,labels = inputs
        return self.generator([latent_vectors,labels])

class GanMonitor(keras.callbacks.Callback):
    def __init__(self, val_dataset, n_samples, epoch_interval=1):
        self.val_images = next(iter(val_dataset))
        self.n_samples = n_samples
        self.epoch_interval = epoch_interval

    def infer(self):
        latent_vector = tf.random.normal(
            shape=(self.model.batch_size, self.model.latent_dim), mean=0.0, stddev=2.0
        )
        return self.model.predict([latent_vector, self.val_images[2]])

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_interval == 0:
            generated_images = self.infer()
            for _ in range(self.n_samples):
                grid_row = min(generated_images.shape[0], 3)
                f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
                for row in range(grid_row):
                    ax = axarr if grid_row == 1 else axarr[row]
                    ax[0].imshow((self.val_images[1][row] + 1) / 2)
                    ax[0].axis("off")
                    ax[0].set_title("Mask", fontsize=20)
                    ax[1].imshow(self.val_images[0][row]/255)
                    ax[1].axis("off")
                    ax[1].set_title("Ground Truth", fontsize=20)
                    ax[2].imshow((generated_images[row] + 1) / 2)
                    ax[2].axis("off")
                    ax[2].set_title("Generated", fontsize=20)
                plt.show()
# %%
gaugan = GauGAN(IMG_HEIGHT,NUM_CLASSES,BATCH_SIZE,latent_dim=256)
gaugan.compile()
#%%
gaugan.fit(train_batches,steps_per_epoch = 100,epochs=20,callbacks=[GanMonitor(train_batches,1)])
# %%
from pathlib import Path

def usingPILandShrink(f,size): 
    im = Image.open(f)  
    im.draft('RGB',(size,size))
    return np.asarray(im)

def load_images(n,path,size):    
    dataset = []
    it = 0
    for filename in Path(path).glob("*.png"):
        if it <= n:
            try:
                im=usingPILandShrink(filename,size)
                im = im.astype('uint8')
                im = cv2.resize(im, dsize=(size,size), interpolation=cv2.INTER_CUBIC)
                im = normalize_maskbis(im)
                dataset.append(im)
            except:
                continue
            it += 1
            if it % 1000==0:
                print(it)
        else:
            break
    return dataset

dataset = load_images(36,"C:/SAMUEL/Centrale/Automatants/Projet_animeface/arcueidtest/annotations/training/",IMG_HEIGHT)
test_dataset = np.asarray(dataset)
print(test_dataset.shape)
#%%
#np.save("unet_test_dataset.npy",test_dataset)
gen = gaugan.generator

##Get style
im = Image.open("C:/SAMUEL/Centrale/Automatants/Projet_animeface/arcueidtest/images/training/240083745_221966026608647_1925481989853930899_n_ccexpress.jpg")
im.draft('RGB',(IMG_HEIGHT,IMG_HEIGHT))
im = np.asarray(im)
im = cv2.resize(im, dsize=(IMG_HEIGHT,IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
im = np.asarray([im])
enc = gaugan.encoder
moy,var = enc(im,256)
style = tf.random.normal((1,256),mean=moy,stddev=var)
##

rand = tf.random.normal((1,256),seed=123)
preds = gen([style,test_dataset])
figure = plt.figure(figsize=(15,15))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.axis('off')
    plt.imshow(preds[i]*0.5+0.5)
plt.show()