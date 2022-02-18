import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from segmentation_pipeline import * 
from IPython.display import clear_output
from losses import * 
from model import * 

batch_size = 4
buffer_size = 100


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
        load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['val'].map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_images
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .map(Augment())
        .map(One_Hot())
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(batch_size).map(One_Hot())

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
u_net = UNET(model)
u_net.compile(keras.optimizers.Adam(learning_rate=4e-4,beta_1=0.5),model_loss=Dice_CE)

def create_mask(pred_mask):
    mask = inv_mask(pred_mask)
    return mask

def show_predictions(dataset=None, num=2):
    if dataset:
        for image, _, true_mask in dataset.take(num):
            pred_mask = u_net.predict(image)
            display([image[0], true_mask[0], create_mask(pred_mask[0])])
    else:
        display([sample_image, sample_mask,
                 create_mask(u_net.predict(sample_image[tf.newaxis, ...]))])


show_predictions(test_batches)

n_epochs = 20
history = u_net.fit(
    train_batches,
    epochs=n_epochs,
    steps_per_epoch=200,
    validation_data=test_batches,
    callbacks=[DisplayCallback(),save_weights()])
