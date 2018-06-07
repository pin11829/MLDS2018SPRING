from __future__ import print_function, division

import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

# import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import os
import sys
import numpy as np

training_data_path = sys.argv[1]
extra_training_data_path = sys.argv[2]

class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        gen_optimizer = keras.optimizers.Adam(lr=0.00015, beta_1=0.5)
        dis_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=dis_optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=gen_optimizer)


    def build_generator(self):

        kernel_init = 'glorot_uniform'

        model = Sequential()

        model.add(Reshape((1,1,100),input_shape=(self.latent_dim,)))
        model.add(Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(0.2))
        model.add(Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(0.2))
        model.add(Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(0.2))
        model.add(Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(0.2))
        model.add(Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        kernel_init = 'glorot_uniform'

        model = Sequential()

        model.add(Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init, input_shape=self.img_shape))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        print('loading the dataset')
        X_train = []

        # faces
        for filename in os.listdir(os.path.join(os.getcwd(),training_data_path)):
            img = io.imread(os.path.join(training_data_path,filename))
            img = resize(img, (self.img_shape), mode='reflect')
            img = img * 2 - 1
            X_train.append(img)
        
        # extra data
        for filename in os.listdir(os.path.join(os.getcwd(),extra_training_data_path)):
            img = io.imread(os.path.join(extra_training_data_path,filename))
            img = img / 127.5 - 1.
            X_train.append(img)

        X_train = np.array(X_train)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        # r, c = 5, 5

        # noise = np.random.normal(0, 1, (r * c, 100))
        # noise = np.random.normal(0, 1, (r*c, 100))
        # gen_imgs = self.generator.predict(noise_z)

        # gen_imgs = 0.5 * gen_imgs + 0.5
        # gen_imgs = (gen_imgs - min(gen_imgs.flatten())) / (max(gen_imgs.flatten())- min(gen_imgs.flatten()))

        # for i in range(r):
        #     tmp = gen_imgs[i*5]
        #     for j in range(1,c):
        #         tmp = np.concatenate((tmp,gen_imgs[i*5+j]))
        #     if i == 0:
        #         plot_imgs = tmp
        #     else:
        #         plot_imgs = np.concatenate((plot_imgs,tmp),axis=1)

        # io.imsave('vis/t_epoch_%d.jpg' % epoch, plot_imgs)

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,:])
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("images/gan_%d.png" % epoch)
        # plt.close()
        
        self.generator.save('gan_model.h5')


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10000, batch_size=64, sample_interval=100)
