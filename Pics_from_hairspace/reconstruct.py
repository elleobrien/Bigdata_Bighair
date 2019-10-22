#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:42:56 2019
This script samples a trained decoder model to query the features represented along each principal axis of the latent space. 
@author: eobrien
"""
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import backend as K
from skimage.transform import resize
from keras.preprocessing.image import img_to_array
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

# Load in the model
# network parameters
original_dim = 128*128
input_shape = (original_dim, )
intermediate_dim = 1024
batch_size = 32
latent_dim = 4

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

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

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# Load weights
vae.load_weights('../../VAE/vae_mlp_hair_4d.h5')



# display a 2D manifold of the digits
n = 50  # figure with 15x15 digits
digit_size = 128
epsilon_std = 0.5
figure = np.zeros((digit_size, digit_size))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)

########### DIMENSION ONE ####################################3
for i, xi in enumerate(grid_x):
    z_sample = np.array([[xi, 0, 0, 0]]) * epsilon_std
    x_decoded = decoder.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap = "gray")
    plt.axis('off')

    fid = "./dim_1/dim_1_" + str(i) + ".png"
    plt.savefig(fid)
    
    
########### DIMENSION TWO ####################################3
for i, xi in enumerate(grid_x):
    z_sample = np.array([[0, xi, 0, 0]]) * epsilon_std
    x_decoded = decoder.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap = "gray")
    plt.axis('off')

    fid = "./dim_2/dim_2_" + str(i) + ".png"
    plt.savefig(fid)

########### DIMENSION THREE ####################################3
for i, xi in enumerate(grid_x):
    z_sample = np.array([[0, 0, xi, 0]]) * epsilon_std
    x_decoded = decoder.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap = "gray")
    plt.axis('off')

    fid = "./dim_3/dim_3_" + str(i) + ".png"
    plt.savefig(fid)

########### DIMENSION FOUR ####################################3
for i, xi in enumerate(grid_x):
    z_sample = np.array([[0, 0, 0, xi]]) * epsilon_std
    x_decoded = decoder.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap = "gray")
    plt.axis('off')

    fid = "./dim_4/dim_4_" + str(i) + ".png"
    plt.savefig(fid)




