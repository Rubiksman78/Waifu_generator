#%%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from segmentation_pipeline import * 
from IPython.display import clear_output
from losses import * 
from model import * 

perso_path = 'C:/SAMUEL/Centrale/Automatants/Waifu_generator/'
batch_size = 4
buffer_size = 500
img_size = 256
num_classes= 7
dataset_path = perso_path + 'segmentation_waifus/images/'

def define_dataset(dataset_path, batch_size, buffer_size):
    training_data = "training/"
    val_data = "validation/"
    TRAINSET_SIZE = len(glob.glob(dataset_path + training_data + "*.jpg"))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    VALSET_SIZE = len(glob.glob(dataset_path + val_data + "*.jpg"))
    print(f"The Validation Dataset contains {VALSET_SIZE} images.")

    train_dataset = tf.data.Dataset.list_files(
        dataset_path + training_data + "*.jpg")
    train_dataset = train_dataset.map(parse_image)

    val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg")
    val_dataset = val_dataset.map(parse_image)
    dataset = {"train": train_dataset, "val": val_dataset}

    train_images = dataset['train'].map(
        lambda x: load_image(x,im_size=img_size), num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['val'].map(
        lambda x: load_image(x,im_size=img_size), num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_images
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .map(Augment())
        .map(One_Hot())
        .repeat(20)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = (
        test_images
        .batch(batch_size)
        .map(Augment())
        .map(One_Hot())
        .repeat(20))
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

train_batches,test_batches = define_dataset(dataset_path,batch_size,buffer_size)

"""
for images, masks ,true_masks in train_batches.take(3):
    sample_image,sample_mask = images[0],inv_mask(masks[0])
    display([sample_image,sample_mask])
"""

arch = u_net_pretrained(num_classes,(img_size,img_size,3))
u_net = UNET(arch)
u_net.compile(
    keras.optimizers.Adam(learning_rate=4e-4,beta_1=0.5),
    model_loss=DiceBCELoss)

#%%
class save_weights(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        self.model.save_weights(perso_path + "segmentation_waifus/u_net_weights.h5")

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=False)
        show_predictions(test_batches,num=1)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def create_mask(pred_mask):
    mask = inv_mask(pred_mask)
    return mask

def show_predictions(dataset=None, num=3):
    if dataset:
        for image, _, true_mask in dataset.take(num):
            pred_mask = u_net.predict(image)
            display([image[0], true_mask[0], create_mask(pred_mask[0])])
    else:
        display([sample_image, sample_mask,
                 create_mask(u_net.predict(sample_image[tf.newaxis, ...]))])

def train(model):
    n_epochs = 20
    history = model.fit(
        train_batches,
        epochs=n_epochs,
        validation_data=test_batches,
        callbacks=[DisplayCallback(),save_weights()])
    return history

#hist = train(u_net)
# %%
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from model import u_net_pretrained
import matplotlib.pyplot as plt

def usingPILandShrink(f,size): 
    im = Image.open(f)  
    im.draft('RGB',(size,size))
    return np.asarray(im)

def load_images(n,path,size):    
    dataset = []
    it = 0
    for filename in Path(path).glob("*.jpg"):
        if it <= n:
            try:
                im=usingPILandShrink(filename,size)
                im = im.astype('uint8')
                im = cv2.resize(im, dsize=(size,size), interpolation=cv2.INTER_CUBIC)
                dataset.append(im)
            except:
                continue
            it += 1
            if it % 1000==0:
                print(it)
        else:
            break
    return dataset

dataset = load_images(4,"D:/Datasets/dataset_92000_256",256)
test_dataset = np.asarray(dataset)
model = u_net.model
model.load_weights(perso_path + 'segmentation_waifus/u_net_weights.h5',
 by_name = True,
 skip_mismatch = True)
preds = u_net.predict(test_dataset)

for i in range(4):
    images = [test_dataset[i],create_mask(np.expand_dims(preds[i],axis=0))]
    figure = plt.figure(figsize=(5,5))
    for j in range(2):
        plt.subplot(1,2,j+1)
        plt.axis('off')
        plt.imshow(images[j])
plt.show()
# %%
