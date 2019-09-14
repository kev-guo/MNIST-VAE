import numpy as np
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential

import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K

#import and load MNIST database
from keras.datasets import mnist
#leave out labels as that are not necessary
(X_train, _), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)
X_test = (X_test.astype(np.float32) - 127.5)/127.5
X_test = X_test.reshape(10000, 784)

#for reproducible results
seed = 1
np.random.seed(seed)

#hyperparameters
epochs = 200
batch_size = 100
starting_dim = 784
hidden_dim = 256
latent_dim = 2

#encoder
e = Input(batch_shape=(batch_size, starting_dim))
hidden_layer = Dense(hidden_dim, activation='relu')(e)
latent_mean = Dense(latent_dim)(hidden_layer)
latent_log_sigma = Dense(latent_dim)(hidden_layer)

#sampling function
def sampling(args):
    z_mean, z_log_sigma = args
    x = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=1.0)
    #reparameterization trick: z=μ+σ⊙x
    return z_mean + K.exp(z_log_sigma) * x

#wrap z as a layer
z = Lambda(sampling)([latent_mean, latent_log_sigma])

#decoder
decoder_hidden_layer = Dense(hidden_dim, activation='relu')
decoder_mean = Dense(starting_dim, activation='sigmoid')
decoded_hidden = decoder_hidden_layer(z)
decoder_output = decoder_mean(decoded_hidden)

#construct VAE, encoder, and generator as functional models
VAE = Model(e, decoder_output)

encoder = Model(e, latent_mean)

gen_start = Input(shape=(latent_dim,))
gen_hidden_layer = decoder_hidden_layer(gen_start)
gen_output = decoder_mean(gen_hidden_layer)
generator = Model(gen_start, gen_output)

def vae_loss(before, after):
    construction_loss = K.sum(K.binary_crossentropy(before, after), axis=1)
    kl_loss = - 0.5 * K.mean(1 + latent_log_sigma - K.square(latent_mean) - K.exp(latent_log_sigma), axis=-1)
    return construction_loss + kl_loss

VAE.compile(optimizer='rmsprop', loss=vae_loss)

VAE.fit(X_train, X_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1)

#plot latent space
points = encoder.predict(X_test, batch_size=batch_size)
plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], c=y_test)
plt.colorbar()
plt.show()

#plot example generated images from decoder
dimensions = 15
image_dimension_size = 28
figure = np.zeros((image_dimension_size * dimensions, image_dimension_size * dimensions))

x_axis = norm.ppf(np.linspace(0.05, 0.95, n)) 
y_axis = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(x_axis):
    for j, xi in enumerate(y_axis):
        latent_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(latent_sample)
        gen_num = x_decoded[0].reshape(image_dimension_size, image_dimension_size)
        figure[i * image_dimension_size: (i + 1) * image_dimension_size,
               j * image_dimension_size: (j + 1) * image_dimension_size]=gen_num

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
