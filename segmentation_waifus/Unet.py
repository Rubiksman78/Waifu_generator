#%%
from email import generator
import tensorflow as tf
from tensorflow import keras
from keras import layers,models,losses
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow_addons as tfa
from pathlib import Path
from PIL import Image
import glob
from pathlib import Path
import cv2

#%%
im_test = np.asarray(cv2.imread(
    "arcueidtest/annotations/validation/10075_result.png"))
L = []
for i in range(len(im_test)):
    for j in range(len(im_test[0])):
        for k in range(3):
            if im_test[i,j,k] not in L:
                L.append(im_test[i,j,k])
print(L)
#%%
#dataset,info = tfds.load('oxford_iiit_pet:3.*.*',with_info=True)
dataset_path = "C:/SAMUEL/Centrale/Automatants/Projet_animeface/arcueidtest/images/"
training_data = "training/"
val_data = "validation/"
TRAINSET_SIZE = len(glob.glob(dataset_path + training_data + "*.jpg"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
VALSET_SIZE = len(glob.glob(dataset_path + val_data + "*.jpg"))
print(f"The Validation Dataset contains {VALSET_SIZE} images.")

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

def inv_mask(mask):
    global palette
    size = mask.shape[1]
    palette2 = tf.constant(palette,dtype=tf.uint8)
    class_indexes = tf.argmax(mask, axis=-1)
    # NOTE this operation flattens class_indexes
    class_indexes = tf.reshape(class_indexes, [-1])
    color_image = tf.gather(palette2, class_indexes)
    color_image = tf.reshape(color_image, [size, size, 3])
    return color_image

def normalize_maskbis(img):
    semantic_map = []
    for colour in palette:
        class_map = tf.reduce_all(tf.equal(img, colour), axis=-1)
        semantic_map.append(class_map)
    semantic_map = tf.stack(semantic_map, axis=-1)
    semantic_map = tf.cast(semantic_map, tf.float32) 
    return semantic_map 
#%%
def parse_image(img_path: str) -> dict:
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

train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + "*.jpg")
train_dataset = train_dataset.map(parse_image)

val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg")
val_dataset =val_dataset.map(parse_image)
dataset = {"train": train_dataset,"val":val_dataset}
#%%
def normalize(input_image):
    input_image = tf.cast(input_image,tf.float32)/255.0
    return input_image

def load_image(datapoint,im_size=128):
    input_image = tf.image.resize(datapoint['image'],(im_size,im_size))
    input_mask = tf.image.resize(datapoint['segmentation_mask'],(im_size,im_size))
    true_mask = tf.image.resize(datapoint['true_mask'],(im_size,im_size))
    input_image = normalize(input_image)
    return input_image,input_mask,true_mask

train_lenght = TRAINSET_SIZE
train_images = dataset['train'].map(load_image,num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['val'].map(load_image,num_parallel_calls=tf.data.AUTOTUNE)
#%%
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=37):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.05,fill_mode = "constant",seed=seed)])
        self.augment_labels = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.05,fill_mode = "constant",seed=seed)])
        self.augment_true_labels = tf.keras.Sequential([layers.RandomFlip(mode="horizontal", seed=seed),layers.RandomRotation(0.05,fill_mode = "constant",seed=seed)])    
    
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

batch_size = 4
buffer_size = 100

train_batches = (
    train_images
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
    .map(Augment())
    .map(One_Hot())
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE))

steps_per_epoch = train_lenght//batch_size
test_batches = test_images.batch(batch_size).map(One_Hot())

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for images, masks ,true_masks in train_batches.take(3):
    sample_image,sample_mask = images[0],inv_mask(masks[0])
    display([sample_image,sample_mask])

#%%
def mobile_net_block(input,filters,strides,kernel_size):
    if strides == 1:
        x = layers.Conv2D(filters,1,padding='same')(input)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
         
        x = layers.DepthwiseConv2D(kernel_size,padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters,1,strides=1,padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        input2 = layers.Conv2D(filters,1,1,padding='same')(input)
        x = layers.Add()([input2,x])
        return x
    else:
        x = layers.Conv2D(filters,1,padding='same')(input)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
         
        x = layers.DepthwiseConv2D(kernel_size,strides=strides,padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters,1,padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        return x
# %%
def define_u_net(classes,img_shape=(64,64,3)):
    init = 'he_normal'
    input = models.Input(shape=img_shape)
    x = layers.Conv2D(32,3,strides=1,padding='same',kernel_initializer=init)(input)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.LeakyReLU()(x)
    res = []
    for filters in [64,128,256,512]:
        x = mobile_net_block(x,filters,1,3)
        res.append(x)
        x = layers.Dropout(0.1)(x)
        x = mobile_net_block(x,filters,2,3)
        x = layers.Dropout(0.1)(x)
    filters_up = [1024,512,256,128]
    for i in range(len(filters_up)):
        x = layers.Conv2D(filters_up[i],3,padding='same',kernel_initializer=init)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv2D(filters_up[i],3,padding='same',kernel_initializer=init)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.UpSampling2D()(x)
        to_add = res[len(res)-i-1]
        to_add = layers.Conv2D(filters_up[i],1,padding='same',kernel_initializer=init)(to_add)
        x = layers.add([to_add,x])

    x = layers.Conv2D(64,3,padding='same',kernel_initializer=init)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.SpatialDropout2D(0.4)(x)
    x = layers.Conv2D(64,3,padding='same',kernel_initializer=init)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.SpatialDropout2D(0.4)(x)
    x = layers.Conv2D(classes,1,activation='softmax')(x)
    model = models.Model(input,x)
    return model

model = define_u_net(7,img_shape=(128,128,3))
# %%
loss_tracker = keras.metrics.Mean(name="loss")
acc_metric = keras.metrics.CategoricalAccuracy(name="accuracy")

class UNET(keras.models.Model):
    def __init__(self,model):
        super(UNET,self).__init__()
        self.model = model

    def compile(self,opt,model_loss):
        super(UNET,self).compile()
        self.opt = opt
        self.loss = model_loss

    @tf.function
    def train_step(self,data):
        images = []
        for x in data:
            images.append(x)
        image,mask,_ = images
        with tf.GradientTape() as tape:
            pred = self.model(image)
            loss = self.loss(mask,pred)
            grad = tape.gradient(loss,self.model.trainable_variables)
            self.opt.apply_gradients(zip(grad,self.model.trainable_variables))
        loss_tracker.update_state(loss)
        acc_metric.update_state(mask,pred)
        return {'loss': loss_tracker.result(),'accuracy':acc_metric.result()}

    def test_step(self,data):
        images = []
        for x in data:
            images.append(x)
        image,mask,_ = images
        pred = self.model(image)
        loss = self.loss(mask,pred)
        loss_tracker.update_state(loss)
        acc_metric.update_state(mask,pred)
        return {'loss': loss_tracker.result(),'accuracy':acc_metric.result()}

    @property
    def metrics(self):
        return [loss_tracker, acc_metric]

    def call(self,inputs):
        return self.model(inputs)

#%%
u_net = UNET(model)
u_net.compile(keras.optimizers.Adam(learning_rate=4e-4,beta_1=0.5),model_loss=losses.CategoricalCrossentropy())
def create_mask(pred_mask):
    mask = inv_mask(pred_mask)
    return mask

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, _ , true_mask in dataset.take(num):
            pred_mask = u_net.predict(image)
            display([image[0],true_mask[0],create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
        create_mask(u_net.predict(sample_image[tf.newaxis, ...]))])

show_predictions(test_batches,num=2)
#%%
from IPython.display import clear_output

class save_weights(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        self.model.save_weights("u_net_weights.h5")

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(test_batches,num=1)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

n_epochs = 20
val_subsplits = 1
validation_steps = TRAINSET_SIZE//batch_size//val_subsplits
history = u_net.fit(
    train_batches,
    epochs=n_epochs,
    steps_per_epoch = 200,
    validation_data=test_batches,
    callbacks=[DisplayCallback(),save_weights()])
 # %%
show_predictions(test_batches,num=1)
# %%
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

dataset = load_images(16,"D:/Datasets/dataset_92000_256",128)
test_dataset = np.asarray(dataset)
#%%
#np.save("unet_test_dataset.npy",test_dataset)
u_net.load_weights('u_net_weights.h5')
preds = u_net.predict(test_dataset)
for i in range(16):
    images = [test_dataset[i],create_mask(np.expand_dims(preds[i],axis=0))]
    figure = plt.figure(figsize=(5,5))
    for j in range(2):
        plt.subplot(1,2,j+1)
        plt.axis('off')
        plt.imshow(images[j])
plt.show()
# %%
