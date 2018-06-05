import PIL
import numpy as np
import tensorflow as tf

def output_single_image(image, height, width, path):
    rgbArray = np.zeros([height, width, 3], 'uint8')
    rgbArray[:, :, 0] = (image[:, :, 0]+1)*128
    rgbArray[:, :, 1] = (image[:, :, 1]+1)*128
    rgbArray[:, :, 2] = (image[:, :, 2]+1)*128
    img = PIL.Image.fromarray(rgbArray)
    img.save(path)

def prelu(x):
    with tf.variable_scope(None, default_name='alpha'):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5
    return pos + neg

def g_res_block(image, residual=True, mode='train', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02)):
    conv1 = tf.layers.conv2d(image, 64, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
    conv1 = tf.layers.batch_normalization(conv1, training=(mode=='train'))
    conv1 = prelu(conv1)

    conv2 = tf.layers.conv2d(conv1, 64, [3, 3], strides=[1, 1], padding='SAME', kernel_initializer=initializer)
    conv2 = tf.layers.batch_normalization(conv2, training=(mode=='train'))
    if residual:
        conv2 = conv2 + image
    else:
        conv2 = conv2

    return conv2

def d_block(image, filters, ksize, stride, mode='train', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02)):
    conv1 = tf.layers.conv2d(image, filters, [ksize, ksize], strides=[stride, stride], padding='SAME', kernel_initializer=initializer)
    conv1 = tf.layers.batch_normalization(conv1, training=(mode=='train'))
    conv1 = tf.nn.leaky_relu(conv1)
    
    return conv1

def _phase_shift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

def pixel_shuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([_phase_shift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def _l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def _spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    with tf.variable_scope(None, default_name='u_'):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = _l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = _l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def separable_conv2d_spec_norm(image, out_channels, kernel_size, stride, multiplier, padding, initializer):
    with tf.variable_scope(None, default_name='separable_conv2d'):
        in_channels = image.shape.as_list()[-1]
        conv_w_d = tf.get_variable('depthwise_kernel', [kernel_size, kernel_size, in_channels, multiplier], initializer=initializer)
        conv_w_p = tf.get_variable('pointwise_kernel', [1, 1, multiplier*in_channels, out_channels], initializer=initializer)
        conv_w_d = _spectral_norm(conv_w_d)
        conv_w_p = _spectral_norm(conv_w_p)
    conv = tf.nn.separable_conv2d(image, conv_w_d, conv_w_p, [1, stride, stride, 1], padding)

    return conv

def conv2d_spec_norm(image, out_channels, kernel_size, stride, padding, initializer):
    with tf.variable_scope(None, default_name='conv2d'):
        in_channels = image.shape.as_list()[-1]
        conv_w = tf.get_variable('conv_kernel', [kernel_size, kernel_size, in_channels, out_channels], initializer=initializer)
        conv_b = tf.get_variable('conv_bias', [out_channels], initializer=tf.zeros_initializer())
        conv_w = _spectral_norm(conv_w)
    conv = tf.nn.conv2d(image, conv_w, [1, stride, stride, 1], padding, ) + conv_b

    return conv

def dense_spec_norm(_input, out_dim, initializer):
    with tf.variable_scope(None, default_name='dense'):
        in_dim = _input.shape.as_list()[-1]
        w = tf.get_variable('kernel', [in_dim, out_dim], initializer=initializer)
        b = tf.get_variable('bias', [out_dim], initializer=tf.zeros_initializer())
        w = _spectral_norm(w)
    out = tf.matmul(_input, w) + b

    return out
