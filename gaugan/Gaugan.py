#%%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from glob import glob
from segmentation_pipeline_bis import * 
from gaugan_class import *
#%%
###Load dataset###
perso_path = 'C:/SAMUEL/Centrale/Automatants/Waifu_generator/' #Mettre votre path local vers le repo
dataset_path = perso_path + 'crash_test_gaugan/images/'

BATCH_SIZE =16
IMG_HEIGHT = IMG_WIDTH = 64
NUM_CLASSES = 7
buffer_size = 50

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
        .take(10000)
        .batch(batch_size,drop_remainder=True)
        .map(Augment())
        .map(One_Hot_bis())
        .repeat(1)
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
"""
for segmentation_map, real_image in zip(sample_train_batch[1], sample_train_batch[0]):
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1).set_title("Segmentation Map")
    plt.imshow(segmentation_map/255)
    fig.add_subplot(1, 2, 2).set_title("Real Image")
    plt.imshow((real_image + 1) / 2)
    plt.show()
"""
class save_weights(keras.callbacks.Callback):
    def __init__(self):
        super(save_weights,self).__init__()

    def on_epoch_end(self,epoch,logs=None):
        self.model.generator.save_weights(f'gen_weights.h5')
        self.model.discriminator.save_weights(f'disc_weights.h5')

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
                    ax[0].imshow(self.val_images[1][row]/255)
                    ax[0].axis("off")
                    ax[0].set_title("Mask", fontsize=20)
                    ax[1].imshow(self.val_images[0][row])
                    ax[1].axis("off")
                    ax[1].set_title("Ground Truth", fontsize=20)
                    ax[2].imshow((generated_images[row] + 1) / 2)
                    ax[2].axis("off")
                    ax[2].set_title("Generated", fontsize=20)
                plt.show()
# %%
gaugan = GauGAN(IMG_HEIGHT,NUM_CLASSES,BATCH_SIZE,latent_dim=256)
gaugan.compile()
gaugan.generator.load_weights('gen_weights1.h5')
gaugan.discriminator.load_weights('disc_weights1.h5')
#%%
gaugan.fit(train_batches,epochs=100,callbacks=[GanMonitor(train_batches,1),save_weights()])
# %%
###Test 
test_dataset = tf.keras.utils.image_dataset_from_directory(
  "C:/SAMUEL/Centrale/Automatants/Waifu_generator/segmentation_waifus/annotations/training/",
  labels=None,
  image_size=(64, 64),
  batch_size=4).map(normalize_maskbis)


n_pairs = 10
gen = gaugan.generator

def show_pairs(true_images):
    j = 0
    style = tf.random.normal((4,256))
    for true_image in true_images.take(3):
        preds = gen([style,true_image])
        for image in preds:
            image = (image*0.5+0.5).numpy()
            figure = plt.figure(figsize=(5,5))
            plt.subplot(1,2,1)
            plt.axis('off')
            plt.imshow(image)
        j += 4

show_pairs(test_dataset)
# %%
gaugan.generator.save_weights('gen_weights.h5')
gaugan.discriminator.save_weights('disc_weights.h5')
# %%
