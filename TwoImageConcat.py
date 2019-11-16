#!/usr/bin/env python
# coding: utf-8

keras = None
shot_change_types = ["Non", "Cut", "Dissolve", "Fade"]

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

def train_model(model, batch_size = 16, train_steps = 100, valid_steps = 25, epochs = 10, view=False):
    class_names = ['Non', 'Cut', 'Dissolve', 'Fade']
    (train_data_gen, valid_data_gen) = get_data_gen(batch_size)
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_data_gen if valid_steps > 0 else None,
        validation_steps=valid_steps if valid_steps > 0 else None
    )
    if view:
        view_history(history)
    return

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
        a[x,y] += 1
    print(a)
    
def frame_to_input(frame_dir, input_dir):
    import os
    os.mkdir(input_dir)
    
    frames = os.listdir(frame_dir)
    
    from PIL import Image
    import Generator
    
    for i in range(len(frames) - 1):
        img1 = Image.open(os.path.join(frame_dir, frames[i]))
        img1.resize((224, 224))
        img2 = Image.open(os.path.join(frame_dir, frames[i + 1]))
        img2.resize((224, 224))
        img = Generator.concat2image(img1, img2)
        img.save(os.path.join(input_dir, frames[i]))

def detection(model, test_dir, threshold = 0.1):
    import os
    import numpy as np
    tests = os.listdir(test_dir)
    total = [0, 0, 0, 0]
    for test in tests:
        img = keras.preprocessing.image.load_img(os.path.join(test_dir, test))
        arr = keras.preprocessing.image.img_to_array(img) / 255.
        pred = model.predict(np.expand_dims(arr, axis=0))[0]
        max_index = np.argmax(pred)
        total[max_index] += 1
        
        change_type = shot_change_types[max_index]
        info = test + ":"
        for i in range(4):
            if pred[i] > threshold:
                info += " " + shot_change_types[i] + "=" + str(int(pred[i] * 100))
        print(info)
        
    print(total)
