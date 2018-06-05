import os
import math

import tensorflow as tf
import numpy as np

import util.util as util

def show_trainable_variable(scope):
    variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)]
    num_vars = 0
    for v in variables:
        print(v)
        num_vars += np.prod([dim.value for dim in v.get_shape()])
    print('Total trainable variables ('+scope+'):', num_vars)

def g_block(image, residual=True, mode='train', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02)):
    conv1 = tf.layers.conv2d(image, 64, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
    conv1 = tf.layers.batch_normalization(conv1, training=(mode=='train'))
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.layers.conv2d(conv1, 64, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
    conv2 = tf.layers.batch_normalization(conv2, training=(mode=='train'))
    if residual:
        conv2 = conv2 + image
    else:
        conv2 = conv2

    return conv2

def d_block(image, filters, ksize, stride, mode='train', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02)):
    conv1 = util.separable_conv2d_spec_norm(image, filters, ksize, stride, 1, 'SAME', initializer)
    '''conv1 = tf.layers.separable_conv2d(
        image, 
        filters, 
        ksize, 
        stride, 
        padding='SAME', 
        depthwise_initializer=initializer, 
        pointwise_initializer=initializer
    )'''
    #conv1 = tf.layers.conv2d(image, filters, [ksize, ksize], strides=[stride, stride], padding='SAME', kernel_initializer=initializer)
    conv1 = tf.nn.leaky_relu(conv1)
    conv2 = util.separable_conv2d_spec_norm(conv1, filters, ksize, stride, 1, 'SAME', initializer)
    '''conv2 = tf.layers.separable_conv2d(
        conv1, 
        filters,
        ksize,
        stride,
        padding='SAME',
        depthwise_initializer=initializer, 
        pointwise_initializer=initializer
    )'''
    #conv2 = tf.layers.conv2d(conv1, filters, [ksize, ksize], strides=[stride, stride], padding='SAME', kernel_initializer=initializer)
    conv2 = conv2 + image
    conv2 = tf.nn.leaky_relu(conv2)
    
    return conv2

class Model():
    def __init__(self, 
        noise_dim, 
        cond1_dim, 
        cond2_dim, 
        height, 
        width, 
        batch_size, 
        learning_rate, 
        beta1, 
        beta2, 
        g_lambda, 
        d_lambda, 
        g_blocks,
        tfrecord_list, 
        path_prefix):
        
        self.noise_dim = noise_dim
        self.cond1_dim = cond1_dim
        self.cond2_dim = cond2_dim
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.g_lambda = g_lambda
        self.d_lambda = d_lambda
        self.g_blocks = g_blocks

        self.path_prefix = path_prefix

        self._generate_input_pipeline(tfrecord_list)

        self._build_model()
    
    def _generate_input_pipeline(self, tfrecord_list):
        def _parse(e):
            features = {
                'jpeg_str': tf.FixedLenFeature([], tf.string),
                'hair_int': tf.FixedLenFeature([], tf.int64),
                'eye_int': tf.FixedLenFeature([], tf.int64)
            }
            features = tf.parse_single_example(e, features)
            image = tf.image.decode_jpeg(features['jpeg_str'], 3)
            image = tf.image.resize_images(image, [self.height, self.width], method=tf.image.ResizeMethod.BILINEAR)
            image = (tf.cast(image, tf.float32) / 128.0) - 1.0
            return image, features['hair_int'],features['eye_int']
        dataset = tf.data.TFRecordDataset(tfrecord_list)
        dataset = dataset.map(_parse)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(300, count=None))

        iterator = dataset.make_one_shot_iterator()

        self.real_image, self.real_cond1, self.real_cond2 = iterator.get_next()
        self.real_image = tf.map_fn(lambda x: tf.image.random_flip_left_right(x), self.real_image)
        self.real_image = tf.contrib.image.rotate(
            self.real_image, 
            tf.random_uniform([], minval=-5*math.pi/180, maxval=5*math.pi/180, dtype=tf.float32),
            interpolation='BILINEAR'
        )
    
    def build_generator(batch_size, noise_dim, cond1_dim, cond2_dim, blocks, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02), mode='train'):
        with tf.variable_scope('generator'):
            noise = tf.random_uniform([batch_size, noise_dim], minval=-1.0, maxval=1.0, dtype=tf.float32)
            cond1 = tf.random_uniform([batch_size], minval=0, maxval=cond1_dim, dtype=tf.int32)
            cond2 = tf.random_uniform([batch_size], minval=0, maxval=cond2_dim, dtype=tf.int32)

            cond1_onehot = tf.one_hot(cond1, cond1_dim)
            cond2_onehot = tf.one_hot(cond2, cond2_dim)

            fc1 = tf.concat([noise, cond1_onehot, cond2_onehot], axis=-1)
            fc1 = tf.layers.dense(fc1, 12*12*64, kernel_initializer=initializer)
            fc1 = tf.layers.batch_normalization(fc1, training=(mode=='train'))
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.reshape(fc1, [batch_size, 12, 12, 64])

            for i in range(blocks):
                if i==0:
                    conv1 = g_block(fc1, residual=True, mode=mode)    
                else:
                    conv1 = g_block(conv1, residual=True, mode=mode)
            
            conv2 = conv1
            
            conv3 = tf.layers.conv2d(conv2, 64, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
            conv3 = tf.layers.batch_normalization(conv3, training=(mode=='train'))
            conv3 = tf.nn.relu(conv3)
            conv3 = conv3 + fc1

            conv4 = tf.layers.conv2d(conv3, 256, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
            conv4 = util.pixel_shuffler(conv4, scale=2)
            conv4 = tf.layers.batch_normalization(conv4, training=(mode=='train'))
            conv4 = tf.nn.relu(conv4)

            conv5 = tf.layers.conv2d(conv4, 256, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
            conv5 = util.pixel_shuffler(conv5, scale=2)
            conv5 = tf.layers.batch_normalization(conv5, training=(mode=='train'))
            conv5 = tf.nn.relu(conv5)

            conv6 = tf.layers.conv2d(conv5, 256, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
            conv6 = util.pixel_shuffler(conv6, scale=2)
            conv6 = tf.layers.batch_normalization(conv6, training=(mode=='train'))
            conv6 = tf.nn.relu(conv6)

            conv7 = tf.layers.conv2d(conv6, 3, [9, 9], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
            conv7 = tf.tanh(conv7)

            return conv7, cond1, cond2
    
    def build_discriminator(batch_size, image, cond1_dim, cond2_dim, mode='train', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02)):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            conv1 = util.conv2d_spec_norm(image, 32, 4, 2, 'SAME', initializer)
            #conv1 = tf.layers.conv2d(image, 32, [4, 4], strides=[2, 2], padding='SAME', kernel_initializer=initializer)
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = d_block(conv1, 32, 3, 1, mode=mode)
            conv3 = d_block(conv2, 32, 3, 1, mode=mode)

            conv4 = util.conv2d_spec_norm(conv3, 64, 4, 2, 'SAME', initializer)
            #conv4 = tf.layers.conv2d(conv3, 64, [4, 4], strides=[2, 2], padding='SAME', kernel_initializer=initializer)
            conv4 = tf.nn.leaky_relu(conv4)

            conv5 = conv4
            for i in range(2):
                conv5 = d_block(conv5, 64, 3, 1, mode=mode)
            
            conv6 = util.conv2d_spec_norm(conv5, 128, 4, 2, 'SAME', initializer)
            #conv6 = tf.layers.conv2d(conv5, 128, [4, 4], strides=[2, 2], padding='SAME', kernel_initializer=initializer)
            conv6 = tf.nn.leaky_relu(conv6)

            conv7 = conv6
            for i in range(2):
                conv7 = d_block(conv7, 128, 3, 1, mode=mode)
            conv8 = util.conv2d_spec_norm(conv7, 256, 3, 2, 'SAME', initializer)
            #conv8 = tf.layers.conv2d(conv7, 256, [3, 3], strides=[2, 2], padding='SAME', kernel_initializer=initializer)
            conv8 = tf.nn.leaky_relu(conv8)

            conv9 = conv8
            for i in range(2):
                conv9 = d_block(conv9, 256, 3, 1, mode=mode)
            conv10 = util.conv2d_spec_norm(conv9, 512, 3, 2, 'SAME', initializer)
            #conv10 = tf.layers.conv2d(conv9, 512, [3, 3], strides=[2, 2], padding='SAME', kernel_initializer=initializer)
            conv10 = tf.nn.leaky_relu(conv10)

            conv11 = conv10
            for i in range(2):
                conv11 = d_block(conv11, 512, 3, 1, mode=mode)
            #conv12 = tf.layers.conv2d(conv11, 1024, [3, 3], strides=[2, 2], padding='SAME', kernel_initializer=initializer)
            #conv12 = tf.nn.leaky_relu(conv12)

            conv12 = tf.reshape(conv11, [batch_size, 3*3*512])

            fc2 = tf.layers.dense(conv12, 2, kernel_initializer=initializer)
            fc3 = tf.layers.dense(conv12, cond1_dim, kernel_initializer=initializer)
            fc4 = tf.layers.dense(conv12, cond2_dim, kernel_initializer=initializer)
            return fc2, fc3, fc4
    
    def _build_model(self):
        fake_image, fake_cond1, fake_cond2 = \
            Model.build_generator(self.batch_size, self.noise_dim, self.cond1_dim, self.cond2_dim, self.g_blocks, mode='train')
        real_logits, real_cond1_logits, real_cond2_logits = \
            Model.build_discriminator(self.batch_size, self.real_image, self.cond1_dim, self.cond2_dim, mode='train')
        fake_logits, fake_cond1_logits, fake_cond2_logits = \
            Model.build_discriminator(self.batch_size, fake_image, self.cond1_dim, self.cond2_dim, mode='train')

        self.g_loss_adv = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self.batch_size], dtype=tf.int32), logits=fake_logits)
        )
        self.g_loss_cls = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fake_cond1, logits=fake_cond1_logits) + 
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fake_cond2, logits=fake_cond2_logits)
        )
        self.g_loss = self.g_loss_adv*self.g_lambda + self.g_loss_cls

        self.d_loss_adv_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self.batch_size], dtype=tf.int32), logits=real_logits)
        )
        self.d_loss_adv_fake = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([self.batch_size], dtype=tf.int32), logits=fake_logits)
        )
        self.d_loss_cls_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.real_cond1, logits=real_cond1_logits) + 
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.real_cond2, logits=real_cond2_logits)
        )
        '''self.d_loss_cls_fake = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fake_cond1, logits=fake_cond1_logits) + 
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fake_cond2, logits=fake_cond2_logits)
        )'''
        self.d_loss = (self.d_loss_adv_real + self.d_loss_adv_fake)*self.d_lambda + (self.d_loss_cls_real)

        self.real_score = tf.reduce_mean(tf.nn.softmax(real_logits)[:, 1])
        self.fake_score = tf.reduce_mean(tf.nn.softmax(fake_logits)[:, 1])

        g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2)

        self.g_global_step = tf.get_variable('g_global_step', initializer=tf.constant(0), trainable=False, dtype=tf.int32)
        self.d_global_step = tf.get_variable('d_global_step', initializer=tf.constant(0), trainable=False, dtype=tf.int32)
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False, dtype=tf.int32)
        self.global_step_plus_one = tf.assign_add(self.global_step, 1)

        #self.clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) 
        #    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.g_train_step = g_optimizer.minimize(
                self.g_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'),
                global_step=self.g_global_step
            )
            self.d_train_step = d_optimizer.minimize(
                self.d_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
                global_step=self.d_global_step
            )
        self.initializer = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.summary_writer = None #tf.summary.FileWriter(os.path.join(self.path_prefix, 'log'))

        show_trainable_variable('generator')
        show_trainable_variable('discriminator')

    def train_generator(self, sess):
        loss, loss_adv, _, fake_score, cls_ = sess.run([self.g_loss, self.g_loss_adv, self.g_train_step, self.fake_score, self.g_loss_cls])
        return loss, loss_adv, fake_score, cls_

    def train_discriminator(self, sess):
        loss_real, loss_fake, real_score, fake_score, loss, cls_, _ = sess.run([self.d_loss_adv_real, self.d_loss_adv_fake, self.real_score, self.fake_score, self.d_loss, self.d_loss_cls_real, self.d_train_step])
        return loss_real, loss_fake, real_score, fake_score, loss, cls_
    
    def add_global_step(self, sess):
        sess.run(self.global_step_plus_one)
    
    def get_global_step(self, sess):
        return sess.run(self.global_step)

    def init(self, sess):
        sess.run(self.initializer)
        print('variables initialized')

    def load(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')

    def save(self, sess, path):
        name = os.path.join(path, 'model.ckpt')
        self.saver.save(sess, name, global_step=self.g_global_step)
        print('Model saved')
    
    def add_summary(self, sess, g_loss_adv, d_loss_adv, g_loss_cls, d_loss_cls, g_score, d_score_real, d_score_fake):
        if self.summary_writer is None:
            self.summary_writer = tf.summary.FileWriter(os.path.join(self.path_prefix, 'log'), sess.graph)
        summary = tf.Summary()
        summary.value.add(tag='Loss/Generator Adversarial Loss', simple_value=g_loss_adv)
        summary.value.add(tag='Loss/Discrminator Adversarial Loss', simple_value=d_loss_adv)
        summary.value.add(tag='Loss/Generator Classification Loss', simple_value=g_loss_cls)
        summary.value.add(tag='Loss/Discrminator Classification Loss', simple_value=d_loss_cls)
        summary.value.add(tag='Score/Generator Score', simple_value=g_score)
        summary.value.add(tag='Score/Discrminator Score (real)', simple_value=d_score_real)
        summary.value.add(tag='Score/Discrminator Score (fake)', simple_value=d_score_fake)
        self.summary_writer.add_summary(summary, sess.run(self.global_step))
        self.summary_writer.flush()

class TestModel():
    def __init__(self, batch_size, noise_dim, cond1_dim, cond2_dim, g_blocks):
        self.fake_image, self.cond1, self.cond2 = \
            Model.build_generator(batch_size, noise_dim, cond1_dim, cond2_dim, g_blocks, mode='test')
        self.fake_image_64 = tf.image.resize_images(self.fake_image, [64, 64], method=tf.image.ResizeMethod.BILINEAR)
        self.saver = tf.train.Saver()
    
    def load(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')
    
    def sample_images(self, sess, cond1, cond2):
        images = sess.run(self.fake_image_64, feed_dict={self.cond1: cond1, self.cond2: cond2})
        return images