### WORK IN PROGRESS
import json
import tensorflow as tf
from keras import Model, layers
from keras.optimizers import Adam
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from keras.backend import random_normal
from keras.losses import binary_crossentropy
from keras.metrics import Mean

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, name="sampling"):
        super().__init__(name=name)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(Model):
    def __init__(
        self,
        hidden_dim=16,
        latent_dim=512,
        name="encoder"
    ):
        super().__init__(name=name)

        self.conv1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu", name=f"{name}-conv1", padding="same")
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu", name=f"{name}-conv1", padding="same")

        self.flat = Flatten(name=f"{name}-flat")

        self.linear = Dense(hidden_dim, name=f"{name}-dense")
        self.mean = Dense(latent_dim, name=f"{name}-mean")
        self.var  = Dense(latent_dim, name=f"{name}-variance")
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.linear(x)
        
        z_mean, z_var = self.mean(x), self.var(x)

        return z_mean, z_var


class Decoder(Model):
    def __init__(
        self,
        input_w,
        input_h,
        input_c=3,
        name="decoder"
    ):
        super().__init__(name=name)

        conved_ip_w = input_w//4
        conved_ip_h = input_h//4

        self.linear = Dense(conved_ip_w * conved_ip_h * 64, activation="relu", name=f"{name}-dense-1")
        self.reshaped = Reshape((conved_ip_w, conved_ip_h, 64), name=f"{name}-reshaped")
        self.convt1 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.convt2 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")

        self.op = Conv2DTranspose(input_c, 3, activation="sigmoid", padding="same", name=f"{name}-output")

    def call(self, x):
        x = self.linear(x)
        x = self.reshaped(x)
        x = self.convt1(x)
        x = self.convt2(x)

        return self.op(x)


class CVAE(Model):
    def __init__(
        self,
        input_w,
        input_h,
        input_c=3,
        batch=256,
        latent_dim=512,
        checkpoint_path=None,
        name="cvae"
    ):
        super().__init__(name=name)


        self.optimizer = Adam(learning_rate=1e-5)
        self.batch = batch

        self.encode = Encoder(latent_dim=latent_dim)
        self.decode = Decoder(input_w, input_h, input_c=input_c)

        self.sampling = Sampling()

        self.total_loss_tracker = Mean(name="total_loss")
        self.recon_loss_tracker = Mean(name="reconstruct_loss")
        self.kl_div_loss_tracker = Mean(name="kl_div_loss")

    def call(self, x):
        z_mean, z_var = self.encode(x)
        z = self.sampling([z_mean, z_var])
        return z_mean, z_var, self.decode(z)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, recon = self(data)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    # binary_crossentropy(data, recon), axis=(1, 2)
                    tf.keras.losses.mean_squared_error(data, recon)
                )
            )
            # tf.print(tf.keras.losses.mean_squared_error(data, recon).shape)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_div_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_div_loss_tracker.result(),
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_div_loss_tracker,
        ]
        

    def summary(self):
        print("Encoder: ")
        print(self.encode.summary())
        print("Decoder: ")
        print(self.decode.summary())

class CVAETrainer():
    def __init__(
            self,
            img_w,
            img_h,
            batch_size,
            data_folder=None,
            input_chan=3,
            # hidden_dim=256,
            latent_dim=32,
            checkpoint_path=None
    ):
        gpus = tf.config.list_physical_devices('GPU')
        print(gpus)
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:  # Use the Default Strategy
            strategy = tf.distribute.get_strategy()

        self.img_w = img_w
        self.img_h = img_h
        self.img_c = input_chan

        self.batch = batch_size
        self.checkpoint_path = checkpoint_path
        
        with strategy.scope():
            self.vae = CVAE(self.img_w, self.img_h, input_c=self.img_c, latent_dim=latent_dim, checkpoint_path=checkpoint_path)

        self.optimizer = Adam()
        self.vae.compile(optimizer=self.optimizer)
        
        if data_folder is not None:
            tds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
                data_folder,
                labels=None,
                batch_size=None,
                color_mode= "rgb" if input_chan == 3 else "grayscale",
                image_size=(self.img_w, self.img_h),
                crop_to_aspect_ratio=True
            )

            self.train_ds = tds.batch(self.batch, drop_remainder=True).shuffle(1024).map(self.preproc_batch).cache()

            self.callbacks = []

            if checkpoint_path is not None:
                self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True, save_weights_only=True))


    def summary(self):
        self.vae(
            tf.zeros(shape=(self.batch, self.img_w, self.img_h, self.img_c), dtype=tf.float32)
        )
        self.vae.summary()

    def fit(self, epochs=2):
        self.vae.fit(self.train_ds, epochs=epochs, callbacks=self.callbacks, shuffle=True)

    def preproc_batch(self, d):
        d = tf.cast(d/255., dtype=tf.float32)
        
        batch = self.batch if d.shape[0] is None else d.shape[0]
        idim  = tf.constant([batch, self.img_w, self.img_h, self.img_c], dtype=tf.int32)

        d = tf.reshape(d, idim)
        
        return d
    
    def save_encoder(self, path):
        self.vae.encode.save(path)
    
    def save_decoder(self, path):
        self.vae.decode.save(path)
    
    def encode(self, data):
        return self.vae.encode(data)
    
    def decode(self, data):
        return self.vae.decode(data)
    

def train():
    cvae = CVAETrainer(28, 28, batch_size=1024, latent_dim=32, data_folder="image/mnist-img/trainSample", input_chan=1, checkpoint_path="saved/cvae/checkpoint")
    cvae.summary()

    cvae.fit(epochs=1000)

    cvae.save_encoder("tf/saved/cvae/encoder")
    cvae.save_decoder("tf/saved/cvae/decoder")

def test():
    from PIL import Image, ImageShow
    from numpy import asarray

    encoder = tf.saved_model.load("tf/saved/cvae/encoder")
    decoder = tf.saved_model.load("tf/saved/cvae/decoder")

    img = Image.open("image/mnist-img/testSample/img_27.jpg").convert('L')
    # img = img.resize((192, 256))
    data = asarray(img)

    print(data.shape)

    tfslice = tf.constant(data, dtype=tf.float32)/255.
    tfslice = tf.reshape(tfslice, (1, 28, 28, 1))

    encoded, _ = encoder(tfslice)
    print("Encoded: ",encoded.shape)
    decoded = decoder(encoded)
    decoded = tf.reshape(decoded * 255., shape=(28, 28)).numpy().astype('uint8')
    print("Decoded: ",decoded.shape)
    
    decodedimg = Image.fromarray(decoded, mode="L")
    ImageShow.show(decodedimg)

import sys

args = sys.argv

# 547/547 [==============================] - 8s 15ms/step - loss: 149.4608 - reconstruction_loss: 143.3177 - kl_loss: 6.2151
if len(args) == 1 or args[1] == "train":
    train()
elif args[1] == "test":
    test()
elif args[1] == "save":
    pass
    # save()
else:
    raise Exception("invalid arg!")
