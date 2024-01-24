import tensorflow as tf 
import glop
import matplotlib.pyplot as plt 
import numpy as np
import os
from keras import layers 


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images=(train_images-127.50)/127.50
buffer_size=60000# no of samples
batch_size=256

train_dataset=tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size=256)


def make_generator_model():
    model=tf.Keras.models.Sequential()
    model.add(layers.dense(7*7*256,use_bias=False,input_shape=(100,)))
    model.add(layers.BatchNormalization)
    model.add(layers.leakyReLU)
    model.add(layers.Reshape(7,7,256))
    
    model.add(layers.Reshape(128,(5,5),padding='same',strides=((1,1),)))
    model.add(layers.BatchNormalization)
    model.add(layers.leakyReLU)
    #assert model.output_shape=(None,7,7,128)
    
    model.add(layers.Reshape(64,(5,5),padding='same',strides=((2,2),)))
    model.add(layers.BatchNormalization)
    model.add(layers.leakyReLU)
    
    
    model.add(layers.Reshape(1,(5,5),padding='same',strides=((2,2),)))
    model.add(layers.BatchNormalization)
    model.add(layers.tanh)
    
def make_Discremenator_model():
    model=tf.Keras.sequential()
    model.add(layers.conv2D(64,(5,5),strids=(2,2),padding='same',input_shape(28,28,1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    model.add(layers.conv2D(8,(5,5),strids=(2,2))
    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())

    
    
    
    
    
    
    