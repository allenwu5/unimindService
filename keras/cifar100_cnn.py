# Setup

import os
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100
from keras.utils import np_utils

nb_epoch = 35
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) # Hack to make print flush immediately

def prepare_input_data(X_train, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_test

def prepare_output_data(y_train, y_test):
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return y_train, y_test

# Script

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train, X_test = prepare_input_data(X_train, X_test)
y_train, y_test = prepare_output_data(y_train, y_test)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch, show_accuracy=True,
                        validation_data=(X_test, y_test),
                        nb_worker=1)

# Save model structure into json and weights into h5
json_string = model.to_json()
open('cifar100.json', 'w').write(json_string)
model.save_weights('cifar100.h5', overwrite=True)