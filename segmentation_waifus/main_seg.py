#%%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from segmentation_pipeline import * 
from IPython.display import clear_output
from losses import * 
from deeplabv3 import *
import glob
#%%
perso_path = 'C:/SAMUEL/Centrale/Automatants/Waifu_generator/' #Mettre votre path local vers le repo
batch_size = 2
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
        .repeat(15) #A modifier si vous voulez plus de data augment
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = (
        test_images
        .batch(batch_size)
        .map(One_Hot())
        .repeat(1))
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
#%%
for images, masks ,true_masks in train_batches.take(3):
    sample_image,sample_mask = images[0],inv_mask(masks[0])
    display([sample_image,sample_mask])
#%%
class save_weights(keras.callbacks.Callback):
    def __init__(self):
        super(save_weights,self).__init__()

    def on_epoch_end(self,epoch,logs=None):
        self.model.modele.save_weights(perso_path + "segmentation_waifus/deeplab.h5") #Mettre le nom que vous voulez aux poids

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
            pred_mask = modele(image)
            display([image[0], true_mask[0], create_mask(pred_mask[0])])
    else:
        display([sample_image, sample_mask,
                 create_mask(modele.predict(sample_image[tf.newaxis, ...]))])

#%%
"""
arch = u_net_pretrained(num_classes,(img_size,img_size,3))
modele = UNET(arch)
modele.compile(
    keras.optimizers.Adam(learning_rate=4e-4,beta_1=0.99),
    model_loss=DiceBCELoss)
"""
arch = DeeplabV3Plus(img_size,num_classes)
modele = DeepLabV3(arch)
modele.compile(
    keras.optimizers.Adam(learning_rate=1e-4),
    model_loss=DiceBCELoss) #Loss modifiable

#%%
modele.modele.load_weights('deeplab.h5')
def train(arch):
    n_epochs = 30
    arch.fit(
        train_batches,
        epochs=n_epochs,
        validation_data=test_batches,
        callbacks=[DisplayCallback(),save_weights()])

train(modele)
#%%
modele.modele.load_weights('deeplab.h5')
from sklearn.metrics import confusion_matrix
import seaborn as sns

true_masks = np.argmax(next(iter(test_batches.map(lambda x,y,z : y))),axis=-1).flatten()
preds = modele.predict(test_batches.map(lambda x,y,z : (x,y,z)))
true_preds = np.argmax(preds,axis=-1).flatten()
cm = confusion_matrix(true_preds,true_masks,normalize='all')
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
class_names = ['hair','eyes','clothes','face','skin','background','mouth']

def show_confusion_matrix(matrix, labels):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(matrix)
    N = len(labels)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(N):
        for j in range(N):
            text = ax.text(j, i,matrix[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    plt.show()
    
#show_confusion_matrix(cm, class_names)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
#%%
test_dataset = tf.keras.utils.image_dataset_from_directory(
  "../../Datasets/anime_face/", #Mettre le path du repo où il y a vos images de test
  labels=None,
  image_size=(256, 256),
  batch_size=2,
  )

model = modele.modele
model.load_weights('deeplab.h5') #Mettre le nom des poids que vous avez save
#%%
def show_pairs(true_images):
    j = 0
    for true_image in true_images.take(1): #Mettre le nombre de batchs que vous voulez voir et save
        preds = model.predict(true_image)
        for i,image in enumerate(true_image):
            images = [image.numpy().astype('uint8'),create_mask(np.expand_dims(preds[i],axis=0)).numpy()]
            
            figure = plt.figure(figsize=(5,5))
            plt.subplot(1,2,1)
            plt.axis('off')
            plt.imshow(images[0])
            plt.subplot(1,2,2)
            plt.axis('off')
            plt.imshow(images[1])
            #Décommenter après si vous voulez save les images obtenues pour la suite
            """ 
            im = Image.fromarray(images[0]) 
            im.save(perso_path+f"crash_test_gaugan/images/training/{i+j}.jpg")
            mask = Image.fromarray(images[1])
            mask.save(perso_path+f"crash_test_gaugan/annotations/training/{i+j}.png")
            """
        j += 4
        #plt.show()

show_pairs(test_dataset)
# %%
