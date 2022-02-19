import tensorflow as tf
from tensorflow import keras
from keras import layers,models
import tensorflow_addons as tfa

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

def define_u_net(classes,img_shape):
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
        x = layers.Conv2DTranspose(filters_up[i],3,padding='same',kernel_initializer=init)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv2DTranspose(filters_up[i],3,2,padding='same',kernel_initializer=init)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)
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