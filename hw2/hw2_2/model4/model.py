import tensorflow as tf
import numpy as np

import os

import config as cfg

def dbg_show_all_variable():
    variables = [v for v in tf.trainable_variables()]
    num_vars = 0
    for k in variables:
        print(k)
        num_vars += np.prod([dim.value for dim in k.get_shape()])
    print('Total trainable variables:', num_vars)

class TrainModel():
    def __init__(
        self, 
        input_x_np, 
        input_y_np, 
        input_x_len_np, 
        input_y_len_np, 
        input_y_shift,
        input_max_len, 
        dict_size, 
        rnn_size, 
        rnn_layer, 
        schedule_rate, 
        learning_rate, 
        learning_rate_decay_steps,
        learning_rate_decay_rate,
        max_grad_norm,
        lstm_keep_prob,
        batch_size):
        with tf.variable_scope('seq2seq'):
            # make dataset
            self.input_x_ph = tf.placeholder(input_x_np.dtype, input_x_np.shape)
            self.input_y_ph = tf.placeholder(input_y_np.dtype, input_y_np.shape)
            self.input_x_len_ph = tf.placeholder(input_x_len_np.dtype, input_x_len_np.shape)
            self.input_y_len_ph = tf.placeholder(input_y_len_np.dtype, input_y_len_np.shape)
            self.input_y_shift_ph = tf.placeholder(input_y_shift.dtype, input_y_shift.shape)

            dataset = tf.data.Dataset.from_tensor_slices((
                self.input_x_ph, 
                self.input_y_ph, 
                self.input_x_len_ph, 
                self.input_y_len_ph,
                self.input_y_shift_ph
            ))
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(3000, count=None))
            
            self.iterator = dataset.make_initializable_iterator()

            self.input_x, self.input_y, self.input_x_len, self.input_y_len, self.input_y_shift = self.iterator.get_next()

            self.lstm_keep_prob = lstm_keep_prob

            def lstm_cell(kp, size):
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=kp)
                return cell

            encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size) for _ in range(rnn_layer)])
            encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size) for _ in range(rnn_layer)])

            embedding = tf.get_variable('embedding', [dict_size, rnn_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_x = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_y = tf.nn.embedding_lookup(embedding, self.input_y)

            encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw, 
                encoder_cell_bw, 
                embedding_x, 
                sequence_length=self.input_x_len, 
                dtype=tf.float32
            )
            encoder_output = tf.concat(encoder_output, -1)

            encoder_states_ = []
            for i in range(rnn_layer):
                encoder_states_.append(
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.concat([encoder_states[0][i].c, encoder_states[1][i].c], axis=-1),
                        tf.concat([encoder_states[0][i].h, encoder_states[1][i].h], axis=-1)
                    )
                )
            encoder_states = tuple(encoder_states_)

            '''encoder_states_ = []
            for i in range(rnn_layer//2, rnn_layer, 1):
                encoder_states_.append(
                    encoder_states[0][i]
                )
                encoder_states_.append(
                    encoder_states[1][i]
                )
            encoder_states = tuple(encoder_states_)'''

            print(encoder_output)
            print(encoder_states)
            print()

            decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size*2) for _ in range(rnn_layer)])

            attent = tf.contrib.seq2seq.LuongAttention(rnn_size*2, encoder_output, memory_sequence_length=self.input_x_len, scale=True)
            attent_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attent, attention_layer_size=rnn_size*2)

            helper = tf.contrib.seq2seq.TrainingHelper(
                embedding_y,
                sequence_length=[input_max_len]*batch_size
            )

            projection_layer = tf.layers.Dense(dict_size, use_bias=False)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                attent_cell,
                helper,
                attent_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states),
                output_layer=projection_layer
            )
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output

            mask = tf.sequence_mask(self.input_y_len, input_max_len, dtype=tf.float32)

            self.train_output = tf.argmax(logits, axis=-1)

            self.loss = tf.reduce_mean(
                tf.reduce_sum(
                    mask * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_shift, logits=logits),
                    axis=-1
                )
            )

            self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False, dtype=tf.int32)

            '''self.learning_rate = tf.train.exponential_decay(
                learning_rate,
                self.global_step,
                learning_rate_decay_steps,
                learning_rate_decay_rate,
                staircase=True
            )'''
            '''self.learning_rate = tf.train.piecewise_constant(
                self.global_step, 
                [20000*5, 20000*6, 20000*7, 20000*8, 20000*9],
                [1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/16, 1.0/32]
            )'''
            self.learning_rate = tf.constant(learning_rate)

            trainer = tf.train.AdamOptimizer(self.learning_rate)
            gradient = tf.gradients(self.loss, tf.trainable_variables())
            clipped_gradient, self.global_norm = tf.clip_by_global_norm(gradient, max_grad_norm)
            self.train_step = trainer.apply_gradients(zip(clipped_gradient, tf.trainable_variables()), global_step=self.global_step)

            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        dbg_show_all_variable()
    
    def get_global_step(self, sess):
        step = sess.run(self.global_step)
        return step
    
    def get_learning_rate(self, sess):
        lr = sess.run(self.learning_rate)
        return lr

    def prepare_dataset(self, sess, input_x, input_y, input_x_len, input_y_len, input_y_shift):
        sess.run(self.iterator.initializer, feed_dict={
            self.input_x_ph: input_x,
            self.input_y_ph: input_y,
            self.input_x_len_ph: input_x_len,
            self.input_y_len_ph: input_y_len,
            self.input_y_shift_ph: input_y_shift
        })
        print('Dataset initialized')
    
    def train(self, sess):
        loss, gn, _= sess.run([self.loss, self.global_norm, self.train_step])
        return loss, gn

    def init(self, sess):
        sess.run(self.initializer)
        print('Variables initialized')

    def save(self, sess, name='model'):
        name = os.path.join(name, 'model.ckpt') # TODO: rename
        self.saver.save(sess, name, global_step=self.global_step)
        print('Model saved')

    def load(self, sess, name='model0'):
        ckpt = tf.train.get_checkpoint_state(name)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')


class EvalModel():
    def __init__(self, input_max_len, dict_size, rnn_size, rnn_layer, batch_size):
        with tf.variable_scope('seq2seq'):
            self.input_x = tf.placeholder(tf.int32, [None, input_max_len])
            self.input_y = tf.placeholder(tf.int32, [None, input_max_len])
            self.input_x_len = tf.placeholder(tf.int32, [None])
            self.input_y_len = tf.placeholder(tf.int32, [None])
            self.input_y_shift = tf.placeholder(tf.int32, [None, input_max_len])

            self.lstm_keep_prob = tf.placeholder(tf.float32)

            def lstm_cell(kp, size):
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=kp)
                return cell

            encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size) for _ in range(rnn_layer)])
            encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size) for _ in range(rnn_layer)])

            embedding = tf.get_variable('embedding', [dict_size, rnn_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_x = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_y = tf.nn.embedding_lookup(embedding, self.input_y)

            encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw, 
                encoder_cell_bw, 
                embedding_x, 
                sequence_length=self.input_x_len, 
                dtype=tf.float32
            )
            encoder_output = tf.concat(encoder_output, -1)

            encoder_states_ = []
            for i in range(rnn_layer):
                encoder_states_.append(
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.concat([encoder_states[0][i].c, encoder_states[1][i].c], axis=-1),
                        tf.concat([encoder_states[0][i].h, encoder_states[1][i].h], axis=-1)
                    )
                )
            encoder_states = tuple(encoder_states_)

            decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size*2) for _ in range(rnn_layer)])

            attent = tf.contrib.seq2seq.LuongAttention(rnn_size*2, encoder_output, memory_sequence_length=self.input_x_len, scale=True)
            attent_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attent, attention_layer_size=rnn_size*2)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding,
                [2]*batch_size,
                3
            )
            helper2 = tf.contrib.seq2seq.TrainingHelper(
                embedding_y,
                sequence_length=[input_max_len]*batch_size
            )

            projection_layer = tf.layers.Dense(dict_size, use_bias=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                attent_cell,
                helper,
                attent_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states),
                output_layer=projection_layer
            )
            decoder2 = tf.contrib.seq2seq.BasicDecoder(
                attent_cell,
                helper2,
                attent_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states),
                output_layer=projection_layer
            )
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=input_max_len)
            outputs2, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder2, maximum_iterations=input_max_len)
            logits = outputs.rnn_output
            logits2 = outputs2.rnn_output

            mask = tf.sequence_mask(self.input_x_len, input_max_len, dtype=tf.float32)

            self.greedy_output = tf.argmax(logits, axis=-1)
            self.train_output = tf.argmax(logits2, axis=-1)

            self.loss = tf.reduce_mean(
                tf.reduce_sum(
                    mask * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_shift, logits=logits2),
                    axis=-1
                )
            )

            self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False, dtype=tf.int32)
            self.saver = tf.train.Saver()

        dbg_show_all_variable()

    def eval(self, sess, input_x, input_x_len, input_y, input_y_len, input_y_shift):
        loss, train_output, greedy_output = sess.run([self.loss, self.train_output, self.greedy_output], feed_dict={
            self.input_x: input_x, 
            self.input_x_len:input_x_len, 
            self.input_y:input_y, 
            self.input_y_len:input_y_len, 
            self.input_y_shift: input_y_shift,
            self.lstm_keep_prob:1.0})
        return loss, train_output, greedy_output
    
    def greedy(self, sess, input_x, input_x_len):
        greedy_output = sess.run(self.greedy_output, feed_dict={
            self.input_x:input_x, 
            self.input_x_len:input_x_len,
            self.lstm_keep_prob:1.0
        })
        return greedy_output

    def load(self, sess, name='model0'):
        ckpt = tf.train.get_checkpoint_state(name)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')



class TestModel():
    def __init__(self, input_max_len, dict_size, rnn_size, rnn_layer, batch_size, beam_width):
        with tf.variable_scope('seq2seq', reuse=tf.AUTO_REUSE):
            self.input_x = tf.placeholder(tf.int32, [None, input_max_len])
            self.input_x_len = tf.placeholder(tf.int32, [None])

            self.lstm_keep_prob = tf.placeholder(tf.float32)

            def lstm_cell(kp, size):
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=kp)
                return cell

            encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size) for _ in range(rnn_layer)])
            encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size) for _ in range(rnn_layer)])

            embedding = tf.get_variable('embedding', [dict_size, rnn_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_x = tf.nn.embedding_lookup(embedding, self.input_x)

            encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw, encoder_cell_bw, embedding_x, sequence_length=self.input_x_len, dtype=tf.float32)
            encoder_output = tf.concat(encoder_output, -1)

            encoder_states_ = []
            for i in range(rnn_layer):
                encoder_states_.append(
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.concat([encoder_states[0][i].c, encoder_states[1][i].c], axis=-1),
                        tf.concat([encoder_states[0][i].h, encoder_states[1][i].h], axis=-1)
                    )
                )
            encoder_states = tuple(encoder_states_)

            tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, beam_width)
            tiled_encoder_states = tf.contrib.seq2seq.tile_batch(encoder_states, beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.input_x_len, beam_width)

            decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.lstm_keep_prob, rnn_size*2) for _ in range(rnn_layer)])

            attent = tf.contrib.seq2seq.LuongAttention(
                rnn_size*2, 
                tiled_encoder_output, 
                memory_sequence_length=tiled_sequence_length, 
                scale=True
            )
            attent_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attent, attention_layer_size=rnn_size*2)

            projection_layer = tf.layers.Dense(dict_size, use_bias=False)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                attent_cell,
                embedding,
                tf.fill([batch_size], 2),
                3,
                attent_cell.zero_state(batch_size*beam_width, tf.float32).clone(cell_state=tiled_encoder_states),
                beam_width,
                output_layer=projection_layer
            )

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=input_max_len)
            self.result = tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])

            self.global_step = tf.get_variable('global_step', shape=[], trainable=False, dtype=tf.int32)
            self.saver = tf.train.Saver()

        dbg_show_all_variable()

    def load(self, sess, name='model0'):
        ckpt = tf.train.get_checkpoint_state(name)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')

    def predict(self, sess, input_x, input_x_len):
        res = sess.run(self.result, feed_dict={self.input_x:input_x, self.input_x_len:input_x_len, self.lstm_keep_prob:1.0})
        return res