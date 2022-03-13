import tensorflow as tf
from tensorflow import keras
from keras import layers
from models import *
from losses_gaugan import *

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

    def compile(self,gen_lr=1e-4,disc_lr=1e-4,**kwargs):
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
        gradients = gradient_tape.gradient(total_loss,self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients,self.discriminator.trainable_variables))
        return total_loss

    def train_generator(self,latent_vector,segmentation_map,labels,image,mean,variance):
        self.discriminator.trainable = False
        with tf.GradientTape()as gen_tape:
            real_output = self.discriminator([segmentation_map,image],training=False)
            fake_output,fake_image = self.combined_model([latent_vector,labels,segmentation_map])
            pred = fake_output[-1]
            g_loss = gen_loss(pred)
            kl_loss = self.kl_divergence_coeff * kl_divergence_loss(mean,variance)
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image,fake_image)
            feature_loss = self.feature_loss_coeff * self.feature_matching_loss(real_output,fake_output)
            total_loss = g_loss + kl_loss + vgg_loss + feature_loss
        all_trainable_variables = (self.combined_model.trainable_variables + self.encoder.trainable_variables)
        gradients = gen_tape.gradient(total_loss,all_trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients,all_trainable_variables))
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