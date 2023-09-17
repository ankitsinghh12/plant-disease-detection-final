from __future__ import print_function
import os
import glob
from keras.models import Sequential
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.layers.normalization import BatchNormalization
import numpy as np

import matplotlib.pyplot as plt

train_dir = "../dataset/color/train"
test_dir = "../dataset/color/val"

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('alexnet_scratch_log.csv')


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


batch_size = 128
epochs = 30
nb_train_samples = get_nb_files(train_dir)
num_classes = len(glob.glob(train_dir + "/*"))
nb_val_samples = get_nb_files(test_dir)

# input image dimensions
IM_WIDTH, IM_HEIGHT = 100, 100
input_shape = (IM_WIDTH, IM_HEIGHT, 3)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IM_WIDTH, IM_HEIGHT), batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IM_WIDTH, IM_HEIGHT), batch_size=batch_size)

model = Sequential()
model.add(Conv2D(64, (11, 11), input_shape=input_shape, padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (3, 3)))

model.add(Conv2D(128, (7, 7), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(192, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(12 * 12 * 256, activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes , activation='softmax'))

model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history_train = model.fit_generator(train_generator, nb_epoch=epochs, steps_per_epoch=nb_train_samples // batch_size,
                                    validation_data=test_generator, nb_val_samples=nb_val_samples // batch_size,
                                    class_weight='auto', callbacks=[lr_reducer, early_stopper, csv_logger])
plot_training(history_train)

model.save("../Models/Alex_Net_scratch.h5")
