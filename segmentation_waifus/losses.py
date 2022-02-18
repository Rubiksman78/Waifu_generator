import tensorflow as tf
from tensorflow import keras 

def soft_dice_loss(y_true, y_pred, epsilon=1e-8):    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * tf.reduce_sum(y_pred * y_true, axes)
    denominator = tf.reduce_sum(tf.sqrt(y_pred) + tf.sqrt(y_true), axes)
    return 1 - tf.reduce_mean((numerator + epsilon) / (denominator + epsilon)) 

def DiceBCELoss(targets, inputs, smooth=1e-6):    
    binary_crossentropy = keras.losses.BinaryCrossentropy()
    BCE =  binary_crossentropy(targets, inputs)
    intersection = tf.reduce_sum(targets * inputs)  
    dice_loss = 1 - (2*intersection + smooth) / (tf.reduce_sum(targets) + tf.reduce_sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    return Dice_BCE

def Dice_CE(y_true, y_pred):
    def dice_loss(y_true, y_pred):
      y_pred = tf.math.sigmoid(y_pred)
      numerator = 2 * tf.reduce_sum(y_true * y_pred)
      denominator = tf.reduce_sum(y_true + y_pred)

      return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + soft_dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)