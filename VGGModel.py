#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

keras = None
shot_change_types = ["Non", "Cut", "Dissolve", "Fade"]

def append_name(model, string):
    if hasattr(model, "_name"):
        model._name = model.name + string
    else:
        model.name = model.name + string
    for layer in model.layers:
        if hasattr(layer, "_name"):
            layer._name = layer.name + string
        else:
            layer.name = layer.name + string
    

def get_generator(directory, batch_size, image_num=1):
    idg = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    data_gen = idg.flow_from_directory(directory=directory,
                                       target_size=(224, 224 * image_num),
                                       class_mode='binary',
                                       batch_size=batch_size)
    return data_gen

def layer_mul_max(a, b):
    x = keras.layers.Multiply()([a, b]) # 7*7*512
    x = keras.layers.MaxPooling2D(pool_size=(7, 7))(x) # 1*1*512
    x = keras.layers.Flatten()(x) # 512
    return x

def layer_max_sub(a, b):
    a = keras.layers.MaxPooling2D(pool_size=(7, 7))(a) # 1*1*512
    b = keras.layers.MaxPooling2D(pool_size=(7, 7))(b) # 1*1*512
    y = keras.layers.Subtract()([a, b]) # 1*1*512
    y = keras.layers.Flatten()(y) # 512
    return y

def layer_concat_conv_max(a, b, c, d):
    z = keras.layers.Concatenate()([a, b, c, d]) # 7*7*2048
    z = keras.layers.Conv2D(filters=1024, kernel_size=(1, 1))(z) # 7*7*1024
    z = keras.layers.MaxPooling2D(pool_size=(7, 7))(z) # 1*1*1024
    z = keras.layers.Flatten()(z) # 1024
    return z

class SingleVGG:
    def make_model():
        vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_model.trainable = False

        model = keras.models.Sequential()
        model.add(vgg16_model)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(4, activation='softmax'))
        
        return model

    def generator(directory, batch_size):
        return get_generator(directory, batch_size)
    
    def image_array_to_input(arr):
        return arr

class DoubleVGG:
    def make_model():
        vgg16_a = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_a.trainable = False
        vgg16_b = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_b.trainable = False
        
        append_name(vgg16_b, "_")
        
        ai = vgg16_a.input
        bi = vgg16_b.input
        a = vgg16_a(ai)
        b = vgg16_b(bi)
        
        x = layer_mul_max(a, b) # 512
        x = keras.layers.Dense(256, activation='relu')(x) # 256
        
        y = layer_max_sub(a, b) # 512
        y = keras.layers.Dense(256, activation='relu')(y) # 256
        
        x = keras.layers.Concatenate()([x, y]) # 512
        x = keras.layers.Dense(4, activation='softmax')(x)
        model = keras.models.Model(inputs=[ai, bi], outputs=x)

        return model

    def generator(directory, batch_size):
        gen = get_generator(directory, batch_size, image_num=2)
        while True:
            (X, Y) = next(gen)
            yield (DoubleVGG.image_array_to_input(X), Y)
    
    def image_array_to_input(arr):
        return np.split(arr, 2, axis=2)

class QuadVGG:
    def make_model():
        vgg16_a = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_a.trainable = False
        vgg16_b = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_b.trainable = False
        vgg16_c = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_c.trainable = False
        vgg16_d = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_d.trainable = False
        
        append_name(vgg16_b, "_")
        append_name(vgg16_c, "__")
        append_name(vgg16_d, "___")
        
        ai = vgg16_a.input
        bi = vgg16_b.input
        ci = vgg16_c.input
        di = vgg16_d.input
        a = vgg16_a(ai)
        b = vgg16_b(bi)
        c = vgg16_c(ci)
        d = vgg16_d(di)
        
        x = layer_mul_max(b, c) # 512
        x = keras.layers.Dense(256, activation='relu')(x) # 256
        
        y = layer_max_sub(a, d) # 512
        y = keras.layers.Dense(256, activation='relu')(y) # 256
        
        z = layer_concat_conv_max(a, b, c, d) # 1024
        z = keras.layers.Dense(512, activation='relu')(z) # 512
        
        x = keras.layers.Concatenate()([x, y, z]) # 1024
        x = keras.layers.Dense(4, activation='softmax')(x)
        model = keras.models.Model(inputs=[ai, bi, ci, di], outputs=x)

        return model

    def generator(directory, batch_size):
        gen = get_generator(directory, batch_size, image_num=4)
        while True:
            (X, Y) = next(gen)
            yield (QuadVGG.image_array_to_input(X), Y)
    
    def image_array_to_input(arr):
        return np.split(arr, 4, axis=2)
