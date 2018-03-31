import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers import Input, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Merge, Activation, concatenate, Add, Concatenate
from keras.datasets import mnist
from keras.callbacks import History
from keras import initializers
from keras.utils import np_utils, to_categorical


def Build_model(x_train, y_train, epoch, batchsize):
    model = Sequential()
    model.add(Dense(256, input_dim=784, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                epochs = epoch,
                shuffle=True,
                batch_size = batchsize,
                validation_split=0.2)
    model.save('model_batch{:.4f}.h5'.format(batchsize))
    return model

if __name__ == '__main__':
    epoch = 100
    Loss_train = []
    Accuracy_train = []
    Loss_test = []
    Accuracy_test = []

    print('Loading mnist...')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('Training the model...')
    Model128 = Build_model(x_train, y_train, epoch, 128)
    Model1024 = Build_model(x_train, y_train, epoch, 1024)

    print('Loading the model...')
    Model128 = load_model('model_batch128.h5')
    Model1024 = load_model('model_batch1024.h5')
    Model_m = load_model('model_batch128.h5')
    x = np.linspace(-1., 2, 100)
    for alpha in x:
        p_merge_net = list(map(lambda x: alpha*x[0] + (1-alpha)*x[1], zip(Model128.get_weights(), Model1024.get_weights())))
        Model_m.set_weights(p_merge_net)
        score_train, acc_train = Model_m.evaluate(x_train, y_train, verbose=0)
        score_test, acc_test = Model_m.evaluate(x_test, y_test, verbose=0)
        Accuracy_train.append(acc_train*100)
        Loss_train.append(score_train)
        Accuracy_test.append(acc_test*100)
        Loss_test.append(score_test)
        print('When alpha = {:.4f}, the loss is the  {:.4f}, accuracy is {:.4f}%'.format(alpha, score_test, acc_test*100))

    print('Drawing...')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, Loss_train, 'r', linestyle='--', label = 'train')
    ax1.plot(x, Loss_test, 'r', label = 'test')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Loss', color='r')
    ax1.set_yscale('log')
    ax2.plot(x, Accuracy_train, 'b', linestyle='--', label = 'train')
    ax2.plot(x, Accuracy_test, 'b', label = 'test')
    ax2.set_ylabel('Accuracy', color='b')
    plt.legend(loc = 'upper right')
    plt.savefig('Q1-3-3-1.png')
    plt.show()
