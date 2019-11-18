#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

keras = None
shot_change_types = ["Non", "Cut", "Dissolve", "Fade"]

def get_generator(directory, batch_size, image_num=1):
    idg = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    data_gen = idg.flow_from_directory(directory=directory,
                                       target_size=(224, 224 * image_num),
                                       class_mode='binary',
                                       batch_size=batch_size)
    return data_gen

class SingleVGG:
    def make_model():
        vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        #vgg16_model.summary()
        vgg16_model.trainable = False

        model = keras.models.Sequential()
        model.add(vgg16_model)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(4, activation='softmax'))
        
        #model.summary()

        return model

    def generator(directory, batch_size):
        return get_generator(directory, batch_size)
    
    def image_array_to_input(arr):
        return arr

class DoubleVGG:
    def make_model(): # ADD
        vgg16_a = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_a.trainable = False
        vgg16_b = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg16_b.trainable = False
        
        if hasattr(vgg16_b, "_name"):
            vgg16_b._name = vgg16_b.name + "_"
        else:
            vgg16_b.name = vgg16_b.name + "_"
        for layer in vgg16_b.layers:
            if hasattr(layer, "_name"):
                layer._name = layer.name + "_"
            else:
                layer.name = layer.name + "_"
        
        ai = vgg16_a.input
        bi = vgg16_b.input
        a = vgg16_a(ai)
        b = vgg16_b(bi)
        x = keras.layers.Subtract()([a, b])
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(4096, activation='relu')(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dense(4, activation='softmax')(x)
        model = keras.models.Model(inputs=[ai, bi], outputs=x)
        
        #model.summary()

        return model

    def generator(directory, batch_size):
        gen = get_generator(directory, batch_size, image_num=2)
        while True:
            (X, Y) = next(gen)
            yield (DoubleVGG.image_array_to_input(X), Y)
    
    def image_array_to_input(arr):
        return np.split(arr, 2, axis=2)
