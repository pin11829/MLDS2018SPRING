import tensorflow as tf
import numpy as np

TIMES = 100
EPOCH = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MONTECARLO = 100

def read_data():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    return train_data, train_labels

def main():
    x, y = read_data()
    input_x = tf.placeholder(tf.float32, [None, 784])
    input_y = tf.placeholder(tf.int64, [None])
    
    fc1_w = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))

    fc1 = tf.matmul(input_x, fc1_w)
    sf = tf.nn.softmax(fc1)

    loss = tf.reduce_mean(tf.reduce_sum(-tf.one_hot(input_y, 10)*tf.log(sf+1e-8), axis=-1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(input_y, tf.argmax(fc1, -1)), tf.float32))
    trainer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_step = trainer.minimize(loss)

    norm = tf.norm(tf.gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    loss2 = norm
    trainer2 = tf.train.AdamOptimizer(LEARNING_RATE)
    train_step2 = trainer2.minimize(loss2)

    plot = []

    for t in range(TIMES):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batches = len(x)//BATCH_SIZE
            for epoch in range(EPOCH):
                tmp = list(zip(x, y))
                np.random.shuffle(tmp)
                x, y = list(zip(*tmp))
                for i in range(batches):
                    loss_, _, accu_ = sess.run([loss, train_step, accuracy], feed_dict={input_x:x[i*BATCH_SIZE:(i+1)*BATCH_SIZE], input_y:y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
                    if i%500==0:
                        print('Training:', loss_, accu_)
            for epoch in range(EPOCH):
                tmp = list(zip(x, y))
                np.random.shuffle(tmp)
                x, y = list(zip(*tmp))
                for i in range(batches):
                    loss2_, _, accu_ = sess.run([loss2, train_step2, accuracy], feed_dict={input_x:x[i*BATCH_SIZE:(i+1)*BATCH_SIZE], input_y:y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
                    if i%500==0:
                        print('Norm', loss2_, accu_)
            orig_loss = sess.run(loss, feed_dict={input_x:x[:BATCH_SIZE], input_y:y[:BATCH_SIZE]})
            w = sess.run(fc1_w)
            print(w.shape)
            print(np.mean(w))
            print(np.std(w))
            high = 0
            low = 0
            for i in range(MONTECARLO):
                h = w + np.random.normal(0, 0.01, [784, 10])
                fc1_w_assign = tf.assign(fc1_w, h)
                sess.run(fc1_w_assign)
                loss_ = sess.run(loss, feed_dict={input_x:x[:BATCH_SIZE], input_y:y[:BATCH_SIZE]})
                if i%50==0:
                    print('Test', loss_)
                if loss_>orig_loss:
                    high += 1
                elif loss_<orig_loss:
                    low += 1
            print('Result:', high, low)
            plot.append([high/MONTECARLO, loss_])
    print(plot)
    plot = np.array(plot)
    np.save('data.npy', plot)


if __name__=='__main__':
    main()