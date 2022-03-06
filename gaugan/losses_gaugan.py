import tensorflow as tf
from tensorflow import keras
from keras import losses

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
        #self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0]
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