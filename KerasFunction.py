#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

keras = None
gen_f = None
shot_change_types = ["Non", "Cut", "Dissolve", "Fade"]

def make_model():
    return gen_f.vgg.make_model()

def load_model(filename):
    model = make_model()
    model.load_weights(filename)
    return model

def save_model(filename, model):
    model.save_weights(filename)
    
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
def get_data_gen(batch_size):
    return (gen_f.vgg.generator("train", batch_size), gen_f.vgg.generator("valid", batch_size))

def view_history(history, epochs):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def train_model(model, batch_size = 16, train_steps = 100, valid_steps = 25, epochs = 10, view=False):
    (train_data_gen, valid_data_gen) = get_data_gen(batch_size)
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_data_gen if valid_steps > 0 else None,
        validation_steps=valid_steps if valid_steps > 0 else None
    )
    if view:
        view_history(history, epochs)
    return history

def check_validation(model, data_gen):
    random_valid = next(data_gen)
    a_test = random_valid[1]
    a_pred = model.predict(random_valid[0])
    a_pred = list(map(np.argmax, a_pred))
    
    from sklearn.metrics import classification_report
    print(classification_report(a_test, a_pred))
    
    a = np.zeros((4, 4), np.int)
    for i in range(a_test.size):
        x = int(a_test[i])
        y = int(a_pred[i])
        a[x,y] += 1
    print(a)

def detection(model, test_dir, threshold = 0.1):
    tests = os.listdir(test_dir)
    total = [0, 0, 0, 0]
    for test in tests:
        img = keras.preprocessing.image.load_img(os.path.join(test_dir, test))
        arr = keras.preprocessing.image.img_to_array(img) / 255
        arr = np.expand_dims(arr, axis=0)
        arr = gen_f.vgg.image_array_to_input(arr)
        pred = model.predict(arr)[0]
        max_index = np.argmax(pred)
        total[max_index] += 1
        
        change_type = shot_change_types[max_index]
        info = test + ":"
        for i in range(4):
            if pred[i] > threshold:
                info += " " + shot_change_types[i] + "=" + str(int(pred[i] * 100))
        print(info)
        
    print(total)
    
def single_detection(model, image_path):
    img = keras.preprocessing.image.load_img(image_path)
    arr = keras.preprocessing.image.img_to_array(img) / 255
    arr = np.expand_dims(arr, axis=0)
    arr = gen_f.vgg.image_array_to_input(arr)
    pred = model.predict(arr)[0]
    return pred
    