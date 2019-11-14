#!/usr/bin/env python
# coding: utf-8

def make_model(keras):
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

def load_model(keras, filename):
    model = make_model(keras)
    model.load_weights(filename)
    return model

def save_model(filename, model):
    model.save_weights(filename)
    
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
def get_data_gen(keras, batch_size):
    train_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    valid_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory='train',
                                                               shuffle=True,
                                                               target_size=(224, 224),
                                                               class_mode='binary')
    valid_data_gen = valid_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory='valid',
                                                               target_size=(224, 224),
                                                               class_mode='binary')
    return (train_data_gen, valid_data_gen)

def view_history(history):
    import matplotlib.pyplot as plt

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

def train_model(model, batch_size = 16, train_steps = 100, valid_steps = 25, epochs = 10):
    class_names = ['Non', 'Cut', 'Dissolve', 'Fade']
    (train_data_gen, valid_data_gen) = get_data_gen(batch_size)
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_data_gen,
        validation_steps=valid_steps
    )
    view_history(history)
    return history

def check_validation(model, data_gen):
    import numpy as np

    random_valid = data_gen.next()
    a_test = random_valid[1]
    a_pred = model.predict_classes(random_valid[0])
    
    from sklearn.metrics import classification_report
    print(classification_report(a_test, a_pred))
    
    a = np.zeros((4, 4), np.int)
    for i in range(a_test.size):
        x = int(a_test[i])
        y = int(a_pred[i])
        a[x,y] = a[x,y] + 1
    print(a)

