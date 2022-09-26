import os
import cv2
import csv
import numpy as np
from time import time
import keras
import tensorflow
import matplotlib
from keras import utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.applications import VGG19
from keras.layers import Conv2D, GlobalAvgPool2D, AvgPool2D, MaxPool2D, Flatten, Dense, Softmax, DepthwiseConv2D, BatchNormalization, ReLU
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input,decode_predictions
#%matplotlib inline

input_size = (32, 32)
dir = 'C:/Users/ADMIN/Desktop/VGG19'

def load_data(path, input_size):
    images = []
    labels = []
    cnt = 0
    for folder in os.listdir(os.path.join(path, 'Train')):
        cur_path = os.path.join(path, 'Train', folder)
        for file_name in os.listdir(cur_path):
            image = cv2.imread(os.path.join(cur_path, file_name))
            image = cv2.resize(image, input_size)
            images.append(image)
            labels.append(int(folder))
    return images, labels

images, labels = load_data(dir + "/", input_size=(32, 32))

def split_train_val_test_data(images, labels):

    # Chuẩn hoá dữ liệu images và labels
    images = np.array(images)
    labels = keras.utils.np_utils.to_categorical(labels)

    # Nhào trộn dữ liệu ngẫu nhiên
    randomize = np.arange(len(images))
    np.random.shuffle(randomize)
    X = images[randomize]
    print("X=", X.shape)
    y = labels[randomize]

    # Chia dữ liệu train theo tỷ lệ 60% train và 40% còn lại cho val
    train_size = int(X.shape[0] * 0.6)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = split_train_val_test_data(images, labels)

classes = 7
batch = 32
epochs = 3
learning_rate = 0.0001

def results(model):
    adam = Adam(lr=learning_rate)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    start = time()
    history = model.fit(X_train, y_train, batch_size=batch, epochs=epochs, validation_data=(X_val, y_val), shuffle = True, verbose=1)
    train_time = time() - start

    model.summary()

    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.plot(history.history['accuracy'], label = 'train_accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(history.history['loss'], label = 'train_loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    start = time()
    test_loss, test_acc = model.evaluate(X_val, y_val)
    test_time = time() - start
    print('\nTrain time: ', train_time)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)
    print('Test time: ', test_time)

model = Sequential()
model.add(VGG19(weights='imagenet', include_top=False, input_shape=(32, 32,3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
model.add(Dense(7))
model.add(Softmax())

results(model)

model.save('C:/Users/ADMIN/Desktop/VGG19/vgg19_32.h5')