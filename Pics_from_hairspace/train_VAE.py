# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:13:13 2019

Variational autoencoder working on hair probability maps

# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import argparse
from PIL import Image

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def load_hair_dataset(rowsize,colsize):
    folder1 = "../Datasets/Berkeley_Yearbook/F_hair/*.png"
    folder2 = "../Datasets/Berkeley_Yearbook/M_hair/*.png"
    
    train_files = glob.glob(folder1) + glob.glob(folder2)
    
    print("Working with {0} images".format(len(train_files)))

    # Original Dimensions    
    channels = 1
    dataset = np.ndarray(shape=(len(train_files), rowsize, colsize,channels),
                         dtype=np.float32)
    
    i = 0
    for _file in train_files:
        #img = load_img(_file, grayscale = True)  # this is a PIL image
        img = Image.open(_file)
        #img = img.convert('RGB')
        #img.thumbnail((image_width, image_height))
        # Convert to Numpy Array
        x = img_to_array(img)  
        #x = x.astype('float32') / 255
        x = x / 65535
       # x = (x - 128.0) / 128.0
        image_resized = resize(x, (rowsize, colsize,channels))
        dataset[i] = image_resized
        i += 1
        if i % 1000 == 0:
            print("%d images to array" % i)
    print("All images to array!")
    return(dataset)
    

# MNIST dataset
size = 128
data = load_hair_dataset(size,size)
(x_train, x_test) = train_test_split(data, shuffle=False, test_size = 0.05)

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 1024
batch_size = 16
latent_dim = 3
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', loss = '')
    vae.summary()
   # plot_model(vae,
   #            to_file='vae_mlp.png',
   #            show_shapes=True)

    # Set the callbacks
    filepath="weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only = False, save_weights_only = False, mode = 'auto', period = 1)
    callbacks_list = [checkpoint]
    
    if args.weights:
        vae.load_weights(args.weights)
        # train the autoencoder
    vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            callbacks=callbacks_list)
    vae.save_weights('vae_mlp_hair_3d.h5')
