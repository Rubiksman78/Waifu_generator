import tensorflow as tf
from tensorflow import keras
from keras import layers,models
import tensorflow_addons as tfa

im_size = 512
im_shape = (im_size,im_size,3)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding='same',
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        kernel_regularizer=keras.regularizers.L2()
    )(block_input)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.SpatialDropout2D(0.8)(x)
    return x

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3],dims[-2]))(dspp_input)
    x = convolution_block(x,kernel_size=1,use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3]//x.shape[1],dims[-2]//x.shape[2]),interpolation="bilinear"
    )(x)

    out_1 = convolution_block(dspp_input,kernel_size=1,dilation_rate=1)
    out_6 = convolution_block(dspp_input,kernel_size=3,dilation_rate=6)
    out_12 = convolution_block(dspp_input,kernel_size=3,dilation_rate=12)
    out_18 = convolution_block(dspp_input,kernel_size=3,dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool,out_1,out_6,out_12,out_18])
    output = convolution_block(x,kernel_size=1)
    return output

def DeeplabV3Plus(image_size,num_classes):
    model_input = models.Input(shape=(image_size,image_size,3))
    resnet50 = keras.applications.mobilenet_v2.MobileNetV2(
        weights="imagenet",include_top=False,input_tensor=model_input
    )
    x = resnet50.get_layer("block_6_expand_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size//4//x.shape[1],image_size//4//x.shape[2]),
        interpolation='bilinear',
    )(x)
    input_b = resnet50.get_layer("block_3_expand_relu").output
    input_b = convolution_block(input_b,num_filters=48,kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a,input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size//x.shape[1],image_size//x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes,kernel_size=(1,1),padding="same",activation="softmax")(x)
    return models.Model(model_input,model_output)


###Classe DeepLabV3 pour entraîner tout ça
loss_tracker = keras.metrics.Mean(name="loss")
acc_metric = keras.metrics.CategoricalAccuracy(name="accuracy")

class DeepLabV3(keras.models.Model):
    def __init__(self,modele):
        super(DeepLabV3,self).__init__()
        self.modele = modele

    def compile(self,opt,model_loss):
        super(DeepLabV3,self).compile()
        self.opt = opt
        self.loss = model_loss

    @tf.function
    def train_step(self,data):
        images = []
        for x in data:
            images.append(x)
        image,mask,_ = images
        with tf.GradientTape() as tape:
            pred = self.modele(image)
            loss = self.loss(mask,pred)
            grad = tape.gradient(loss,self.modele.trainable_variables)
            self.opt.apply_gradients(zip(grad,self.modele.trainable_variables))
        loss_tracker.update_state(loss)
        acc_metric.update_state(mask,pred)
        return {'loss': loss_tracker.result(),'accuracy':acc_metric.result()}

    def test_step(self,data):
        images = []
        for x in data:
            images.append(x)
        image,mask,_ = images
        pred = self.modele(image)
        loss = self.loss(mask,pred)
        loss_tracker.update_state(loss)
        acc_metric.update_state(mask,pred)
        return {'loss': loss_tracker.result(),'accuracy':acc_metric.result()}

    @property
    def metrics(self):
        return [loss_tracker, acc_metric]

    def call(self,inputs):
        return self.modele(inputs)