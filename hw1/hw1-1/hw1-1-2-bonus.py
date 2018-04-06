import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import math

from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers import Input, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Merge, Activation, concatenate, Add, Concatenate
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.callbacks import History
from keras import initializers
from keras.utils import np_utils, to_categorical


def Bulid_Shallowmodel(x_train, y_train, epoch):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(32, 32, 3)))
    model.add(Convolution2D(612, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                epochs = epoch,
                shuffle=True,
                batch_size = 128,
                validation_split=0.2)
    model.save('model_Shallowmodel.h5')
    np.save("Shistory.npy", history.history['loss'])
    np.save("Shistory2.npy", history.history['acc'])
    return model, history

def Bulid_Deepmodel(x_train, y_train, epoch):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(32, 32, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(480, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                epochs = epoch,
                shuffle=True,
                batch_size = 256,
                validation_split=0.2)
    model.save('model_Deepmodel.h5')
    np.save("Dhistory.npy", history.history['loss'])
    np.save("Dhistory2.npy", history.history['acc'])
    return model, history

def Bulid_VeryDeepmodel(x_train, y_train, epoch):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(32, 32, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                epochs = epoch,
                shuffle=True,
                batch_size = 256,
                validation_split=0.2)
    model.save('model_VeryDeepmodel.h5')
    np.save("Vhistory.npy", history.history['loss'])
    np.save("Vhistory2.npy", history.history['acc'])
    return model, history

if __name__ == '__main__':
    epoch = 100

    print('Loading cifar10...')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(y_train.shape, 'y_train samples')


    print('Training the model...')
    Shallowmodel, Shistory = Bulid_Shallowmodel(x_train, y_train, epoch)
    Deepmodel, Dhistory = Bulid_Deepmodel(x_train, y_train, epoch)
    VeryDeepmodel, Vhistory = Bulid_VeryDeepmodel(x_train, y_train, epoch)



    print('Loading the model...')
    Shallowmodel = load_model('model_Shallowmodel.h5')
    Deepmodel = load_model('model_Deepmodel.h5')
    VeryDeepmodel = load_model('model_VeryDeepmodel.h5')
    Shistoryloss = np.load("Shistory.npy")
    Dhistoryloss = np.load("Dhistory.npy")
    Vhistoryloss = np.load("Vhistory.npy")
    Shistoryacc = np.load("Shistory2.npy")
    Dhistoryacc = np.load("Dhistory2.npy")
    Vhistoryacc = np.load("Vhistory2.npy")

    print('Drawing the loss...')
    plt.plot(Shistoryloss, 'r', label = 'ShallowModel')
    plt.plot(Dhistoryloss, 'y', label = 'MidModel')
    plt.plot(Vhistoryloss, 'g', label = 'DeepModel')
    plt.legend(loc = 'upper right')
    plt.title('Q1-1-2-2b')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('Q1-1-2-2b.png')
    plt.show()


    print('Drawing the accuracy...')
    plt.plot(Shistoryacc, 'r', label = 'ShallowModel')
    plt.plot(Dhistoryacc, 'y', label = 'MidModel')
    plt.plot(Vhistoryacc, 'g', label = 'DeepModel')
    plt.legend(loc = 'lower right')
    plt.title('Q1-1-2-3b')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('Q1-1-2-3b.png')
    plt.show()
