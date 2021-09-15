# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:04:43 2021

use tf.keras rather than keras package
https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/


@author: James Ang
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from os import path

from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU,\
    Conv2DTranspose, Conv2D, Reshape, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.data import Dataset
import time

import matplotlib.pyplot as plt

# print(tf.version)


def timer(func):
    
    def wrapper():
        
        before = time.time()
        func()
        print("Function takes",time.time()-before,"seconds")

    return wrapper

def make_generator():
    
    generator = Sequential(
            [
                # Input layer
                Input(shape=(100,)),
                
                # 1st layer
                Dense(4*4*256, use_bias=False),
                BatchNormalization(),
                LeakyReLU(),
                Reshape((4,4,256)),
                
                # 2nd layer
                Conv2DTranspose(filters = 128, kernel_size = (4,4), 
                                strides = (1, 1), padding = "same", use_bias=False,
                                data_format="channels_last"),
                BatchNormalization(),
                LeakyReLU(),
              
                # 3rd layer
                Conv2DTranspose(filters = 64, kernel_size = (4,4), 
                                strides = (2, 2), padding = "same", use_bias=False,
                                data_format="channels_last"),
                BatchNormalization(),
                LeakyReLU(),
                
                # 4th layer
                Conv2DTranspose(filters = 32, kernel_size = (4,4), 
                                strides = (2, 2), padding = "same", use_bias=False,
                                data_format="channels_last"),
                BatchNormalization(),
                LeakyReLU(),
                
                # Final layer
                Conv2DTranspose(filters = 3, kernel_size = (4,4), 
                                strides = (2, 2), padding = "same", use_bias=False,
                                data_format="channels_last", activation='tanh'),
            ]
        )
    # Note: kernel size plays a role in the total number of trainable parameters.

    return generator


def make_discriminator():
    
    discriminator = Sequential(
            [   
                # Input layer
                Input(shape=(32,32,3)),
                
                # 1st Layer
                Conv2D(32, (3,3), strides=(1, 1), padding='same', 
                       activation="relu"),
                LeakyReLU(alpha=0.2),
                Dropout(0.3),
                
                # 2nd Layer
                Conv2D(64, (3,3), strides=(2, 2), padding='same', 
                       activation="relu"),
                LeakyReLU(alpha=0.2),
                Dropout(0.3),
                
                # 3rd Layer
                Conv2D(128, (3,3), strides=(2, 2), padding='same', 
                       activation="relu"),
                LeakyReLU(alpha=0.2),
                Dropout(0.3),
                
                # 4th Layer
                Conv2D(256, (3,3), strides=(2, 2), padding='same', 
                       activation="relu"),
                LeakyReLU(alpha=0.2),
                Dropout(0.3),
                
                Flatten(),
                Dense(1, activation = 'sigmoid'),
                
                ]
        )
    
    return discriminator


def see_images(X_train, y_train, class_names):
    
    # Just to see some images
    plt.figure(figsize=(10,10))
    
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i].astype('uint8'), cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[y_train[i][0]])


def see_sample_G_image(generator, z_noise_dim = 100, batch_size = 32):
    
    # GENERATOR
    print(f"Generator output shape is: {generator.output_shape}")
    
    input_z = tf.random.normal([batch_size, z_noise_dim]) # TensorShape([[batch_size, z_noise_dim]])
    
    
    image = generator(input_z, training=False)
    c = np.squeeze(image)* 127.5 + 127.5
    # plt.imshow(c[0],cmap='inferno')
    plt.imshow(c[0].astype('uint8'))
    plt.axis('off')
    
def see_sample_D_decision(discriminator, batch_size = 32):
    
    
    print(f"Discriminator output shape is: {discriminator.output_shape}")
    
    input_test_image = tf.random.normal([batch_size, 32, 32, 3])
    
    decision = discriminator(input_test_image, training=False)
    print(decision)
    
    return decision


def D_loss_func(cross_entropy, real_output, fake_output):
    
    # disc_bce_loss = BinaryCrossentropy(from_logits=True)
    # bce(y_true, y_pred).numpy()
    D_realloss = cross_entropy(tf.ones_like(real_output), real_output)#.numpy()
    D_fakeloss = cross_entropy(tf.zeros_like(fake_output), fake_output)#.numpy()
    total_D_loss = (D_realloss + D_fakeloss)*0.5 # *0.5 for average?
    
    return total_D_loss 


def G_loss_func(cross_entropy, fake_output):
    
    print('enter G_lossfunc')
    # Only when training with generated fake images
    # gen_bce_loss = BinaryCrossentropy(from_logits=True)
    
    # see Ahlad Kumar GAN chapter 4 in evernote for explanation of G_loss
    G_loss = cross_entropy(tf.ones_like(fake_output), fake_output)#.numpy()
    
    return G_loss


# Use train step just to handle the batch size
@tf.function
def train_step(batch_size, z_noise_dim, real_images, generator, discriminator, gen_optimiser, disc_optimiser):
    print('enter train_step')
    input_z = tf.random.normal([batch_size, z_noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        print('enter gradienttape')
        fake_images = generator(input_z, training=True)
        
        fake_output = discriminator(fake_images, training = True)
        real_output = discriminator(real_images, training = True)
        
        # Loss function
        cross_entropy = BinaryCrossentropy(from_logits=True)
        
        G_loss = G_loss_func(cross_entropy, fake_output)
        
        total_D_loss = D_loss_func(cross_entropy, real_output, fake_output)
        
    gen_gradient = gen_tape.gradient(G_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(total_D_loss, discriminator.trainable_variables )

    gen_optimiser.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    disc_optimiser.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))


def train(dataset, epochs, batch_size, z_noise_dim, generator, discriminator, gen_optimiser, disc_optimiser):
    
    for epoch in range(epochs):
        
        start_time = time.time()
        
        print("Epoch", epoch)
        i = 0
        for real_images in dataset:
            
            train_step(batch_size, z_noise_dim, real_images, generator, discriminator, gen_optimiser, disc_optimiser)
            i += 1
            print(i)
            # see_sample_G_image(generator, z_noise_dim, batch_size)
            
        print("Time for epoch {} is {}." .format(epoch + 1, time.time() - start_time))
        
        generate_and_save_images(generator, epoch, z_noise_dim)

# GENERATE AND SAVE IMAGES
def generate_and_save_images(model, epoch, z_noise_dim):
    
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, z_noise_dim]) # TensorShape([16, 100])
    
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
        
    predictions = model(seed, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        c = np.squeeze(predictions[i, :, :, :]) * 127.5 + 127.5
        plt.imshow(c.astype('uint8'))
        # plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5)
        plt.axis('off')
    
    # Generate and Save Images
    save_image_dir = "gan2"
    
    if path.exists(save_image_dir):
        
        print('directory exists')
        
    else:
        os.mkdir(save_image_dir)
        
    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(save_image_dir,epoch))
    plt.close()

def main():
    
    # Import Data--------------------------------------------------------------------
    
    cifar10 = tf.keras.datasets.cifar10
    
    (X_train, y_train), (_, _) = cifar10.load_data()
    assert X_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # See images
    see_image_flag = 0
    
    if see_image_flag:
        see_images(X_train, y_train, class_names)
        
        
    # NORMALISE DATASET IMAGES to [-1 , 1]-----------------------------------------------
    
    X_train = (X_train - 127.5) / 127.5
    
    # BATCH AND SHUFFLE THE DATA---------------------------------------------------------
    
    epochs = 2
    batch_size = 128
    z_noise_dim = 100
    buffer_size = len(X_train) # Used for Shuffling. 
    
    # 'from_tensor_slices' - Get the slices of an array in the form of objects
    # Refer here - https://www.geeksforgeeks.org/tensorflow-tf-data-dataset-from_tensor_slices/
    
    dataset = Dataset.from_tensor_slices(X_train).shuffle(buffer_size).batch(batch_size)
    
    # for elem in dataset:
    #     print(elem.shape)
    
    
    
    
    
    # CREATE GENERATOR---------------------------------------------------------------------
    
    generator = make_generator()
    generator.summary()
    

    # Test Untrained-Genarator
    see_sample_G = 0
    
    if see_sample_G:
        see_sample_G_image(generator, z_noise_dim, batch_size)
    

    # CREATE DISCRIMINATOR-----------------------------------------------------------------
    
    discriminator = make_discriminator()
    discriminator.summary()
    
    # Test Untrained-Discriminator
    see_decision_D = 0
    
    if see_decision_D:
        see_sample_D_decision(discriminator, batch_size)
    
    # Optimisers
    gen_optimiser = optimizers.Adam(learning_rate=0.001)
    disc_optimiser = optimizers.Adam(learning_rate=0.001)
    
    # Train
    train(dataset, epochs, batch_size, z_noise_dim, generator, discriminator, gen_optimiser, disc_optimiser)

    

if __name__ == '__main__':
    main()