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
from keras.callbacks import History
from keras import initializers



def Bulid_Shallowmodel(x_train, y_train, epoch):
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(Dense(280, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss = 'mse', optimizer = 'Adam')
    history = model.fit(x_train, y_train,
                epochs = epoch,
                batch_size = 1,
                shuffle = True)
    model.save('model_Shallowmodel.h5')
    np.save("Shistory.npy", history.history['loss'])
    return model, history

def Bulid_Deepmodel(x_train, y_train, epoch):
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss = 'mse', optimizer = 'Adam')
    history = model.fit(x_train, y_train,
                epochs = epoch,
                shuffle=True)
    model.save('model_Deepmodel.h5')
    np.save("Dhistory.npy", history.history['loss'])
    return model, history

def Bulid_VeryDeepmodel(x_train, y_train, epoch):
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss = 'mse', optimizer = 'Adam')
    history = model.fit(x_train, y_train,
                epochs = epoch,
                shuffle=True)
    model.save('model_VeryDeepmodel.h5')
    np.save("Vhistory.npy", history.history['loss'])
    return model, history



if __name__ == '__main__':
    epoch = 10000

    print('Drawing the function...')
    x = np.linspace(0.1, 0.5, 2000)
    y = np.sin(10 * np.pi * x) / (10 * np.pi * x)
    plt.plot(x, y, 'b', label='Ground-truth')

    print('Training the model...')
    Shallowmodel, Shistory = Bulid_Shallowmodel(x, y, epoch)
    Deepmodel, Dhistory = Bulid_Deepmodel(x, y, epoch)
    VeryDeepmodel, Vhistory = Bulid_VeryDeepmodel(x, y, epoch)

    print('Loading the model...')
    Shallowmodel = load_model('model_Shallowmodel.h5')
    Deepmodel = load_model('model_Deepmodel.h5')
    VeryDeepmodel = load_model('model_VeryDeepmodel.h5')
    Shistory = np.load("Shistory.npy")
    Dhistory = np.load("Dhistory.npy")
    Vhistory = np.load("Vhistory.npy")

    print('Predicting the data...')
    y_predS = Shallowmodel.predict(x)
    y_predD = Deepmodel.predict(x)
    y_predV = VeryDeepmodel.predict(x)

    print('Drawing the curve...')
    plt.plot(x, y_predS, 'r', label = 'ShallowModel')
    plt.plot(x, y_predD, 'y', label = 'MidModel')
    plt.plot(x, y_predV, 'g', label = 'DeepModel')
    plt.legend(loc = 'lower right')
    plt.title('Q1-1-3')
    plt.show()

    print('Drawing the loss...')
    plt.plot(Shistory, 'r', label = 'ShallowModel')
    plt.plot(Dhistory, 'y', label = 'MidModel')
    plt.plot(Vhistory, 'g', label = 'DeepModel')
    plt.legend(loc = 'upper right')
    plt.title('Q1-1-2')
    plt.yscale('log')
    plt.show()
