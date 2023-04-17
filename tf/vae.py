import tensorflow as tf

from keras import Model
from keras.optimizers import Adam
from keras.layers import Dense, LeakyReLU
from keras.losses import MeanSquaredError
from keras.metrics import Mean

class Encoder(Model):
    def __init__(
            self,
            hidden_dim=256,
            latent_dim=32,
            name="vae-encoder"
    ):
        super().__init__(name=name)

        self.leaky = LeakyReLU()
        self.hidden = Dense(hidden_dim, name="encoder-hidden", activation=self.leaky)
        

        self.mean = Dense(latent_dim, name="encoder-mean")
        self.variance = Dense(latent_dim, name="encoder-variance")

    def call(self, x):
        x = self.hidden(x)
        # x = self.leaky(x)

        z_mean = self.mean(x)
        z_variance = self.variance(x)

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_variance) * epsilon, z_mean, z_variance
    
class Decoder(Model):
    def __init__(
        self,
        origin_dim,
        hidden_dim=256,
        name="vae-decoder"
    ):
        super().__init__(name=name)
        
        self.leaky = LeakyReLU()

        self.hidden = Dense(hidden_dim, activation=self.leaky)

        self.op = Dense(origin_dim)

    def call(self, x):
        x = self.hidden(x)
        # x = self.leaky(x)

        return self.op(x)
        

class VAE(Model):
    def __init__(
            self,
            input_dim,
            batch=64,
            hidden_dim=256,
            latent_dim=32, # the final input - output to bottleneck
            checkpoint_path=None,
            name="vae"
    ):
        super().__init__(name=name)

        self.batch = batch

        self.encode = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decode = Decoder(input_dim, hidden_dim=hidden_dim)

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruct_loss_tracker = Mean(name="reconstruct_loss")
        self.kl_div_loss_tracker = Mean(name="kl_div_loss")

        self.mse_loss = MeanSquaredError()

        self.callbacks = []

        if checkpoint_path is not None:
            try:
                self.load_weights(checkpoint_path)
            except:
                print("Model checkpoint not found!")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruct_loss_tracker,
            self.kl_div_loss_tracker,
        ]
    
    def call(self, x):
        z, mu, sigma = self.encode(x)

        return self.decode(z), mu, sigma
    
    def summary(self):
        print("Encoder ------------------------")
        print(self.encode.summary())
        print("Decoder ------------------------")
        print(self.decode.summary())

    def train_step(self, x):
        with tf.GradientTape() as tape:
            recon, mean, var = self(x)
            
            recon_loss = self.mse_loss(x, recon)
            kl_loss = -0.5 * (1 + var - tf.square(mean) - tf.exp(var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruct_loss_tracker.update_state(recon_loss)
        self.kl_div_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruct_loss_tracker.result(),
            "kl_loss": self.kl_div_loss_tracker.result(),
        }

class VAETrainer():
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
            self.vae = VAE(self.input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, checkpoint_path=checkpoint_path)

        self.optimizer = Adam(learning_rate=2e-6)
        self.vae.compile(optimizer=self.optimizer)
        
        if data_folder is not None:
            tds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
                data_folder,
                labels=None,
                batch_size=None,
                color_mode="rgb",
                image_size=(self.img_w, self.img_h),
                crop_to_aspect_ratio=True
            )

            self.train_ds = tds.batch(self.batch, drop_remainder=True).shuffle(1024).map(self.preproc_batch)

            self.callbacks = []

            if checkpoint_path is not None:
                self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True, save_weights_only=True))


    def summary(self):
        self.vae(
            tf.zeros(shape=(self.batch, self.input_dim))
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
    trainer = VAETrainer(
        192,
        256,
        512,
        "image",
        hidden_dim=1024,
        latent_dim=512,
        checkpoint_path="tf/saved/checkpoint/cp.ckpt"
    )

    trainer.summary()
    trainer.fit(epochs=100)

    save()

def save():
    trainer = VAETrainer(
        192,
        256,
        512,
        "image",
        hidden_dim=1024,
        latent_dim=512,
        checkpoint_path="tf/saved/checkpoint/cp.ckpt"
    )

    trainer.summary()

    trainer.save_encoder("tf/saved/encoder")
    trainer.save_decoder("tf/saved/decoder")

# lets test an encoding
def test():
    from PIL import Image, ImageShow
    from numpy import asarray

    encoder = tf.saved_model.load("tf/saved/encoder")
    decoder = tf.saved_model.load("tf/saved/decoder")

    img = Image.open("image/019595_0.jpg")
    data = asarray(img)

    tfslice = tf.constant(data, dtype=tf.float32)/255.
    tfslice = tf.reshape(tfslice, (1, 192 * 256 * 3))

    encoded, _, _ = encoder(tfslice)
    print("Encoded: ",encoded.shape)
    decoded = decoder(encoded)
    decoded = tf.reshape(decoded * 255., shape=(192, 256, 3)).numpy().astype('uint8')
    print("Decoded: ",decoded.shape, decoded)
    
    decodedimg = Image.fromarray(decoded)
    ImageShow.show(decodedimg)

test()
# train()
# save()