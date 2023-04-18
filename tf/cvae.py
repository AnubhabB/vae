### WORK IN PROGRESS

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.layers import LeakyReLU, Conv2D, Flatten, Dense

class Encoder(Model):
    def __init__(
        self,
        latent_dim=512,
        name="encoder"
    ):
        super().__init__(name=name)

        self.conv1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu", name=f"{name}-conv1")
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu", name=f"{name}-conv1")

        self.flat = Flatten(name=f"{name}-flat")

        self.mean = Dense(latent_dim, name=f"{name}-mean")
        self.var  = Dense(latent_dim, name=f"{name}-variance")
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        
        z_mean, z_var = self.mean(x), self.var(x)

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_var) * epsilon, z_mean, z_var

    

class Decoder(Model):
    def __init__(
        self,
        input_w,
        input_h,
        input_c=3,
        latent_dim=512,
        name="decoder"
    ):
        super().__init__(name=name)

        self.dense1 = Dense()
        pass

    def call(self, x):
        pass


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


        self.optimizer = Adam(learning_rate=1e-6)
        self.batch = batch

        self.encode = Encoder(latent_dim=latent_dim)
        self.decode = Decoder(input_w, input_h, latent_dim=latent_dim)

    def call(self, x):
        z, z_mean, z_variance = self.encode(x)

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
            hidden_dim=256,
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
        self.input_dim = img_w * img_h * input_chan

        self.batch = batch_size
        self.checkpoint_path = checkpoint_path
        
        with strategy.scope():
            self.vae = CVAE(img_w, img_h, latent_dim=latent_dim, checkpoint_path=checkpoint_path)

        self.optimizer = Adam(learning_rate=2e-6)
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
            tf.zeros(shape=(self.batch, self.img_w, self.img_h, self.img_c))
        )
        self.vae.summary()

    def fit(self, epochs=2):
        self.vae.fit(self.train_ds, epochs=epochs, callbacks=self.callbacks, shuffle=True)

    def preproc_batch(self, d):
        d = d/255.
        print(d.shape)
        batch = self.batch if d.shape[0] is None else d.shape[0]
        idim  = tf.constant([batch, self.input_dim], dtype=tf.int32)

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
    cvae = CVAETrainer(28, 28, batch_size=512, data_folder="image/mnist-img/trainSample", input_chan=3)
    cvae.summary()


train()