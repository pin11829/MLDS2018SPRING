import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from keras.preprocessing import sequence
from tensorflow.python.layers.core import Dense
import random
import json
import argparse

word_threshold = 3
dim_image = 4096
dim_hidden = 256
video_lstm_step = 80
caption_lstm_step = 20
learning_rate = 0.001
max_gradient_norm = 5
epochs = 90
batch_size = 50
learning_rate = 0.001

training_feat_path = './training_data/feat'
datapath = './MLDS_hw2_1_data'
model_save_dir = './model_s2s'
model_path='./model_s2s/model-200'

def Remove_redundent(captions):
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)
    return captions

def Bulid_Wordvocab(total_labels, word_count_threshold):
    print('Counting the vocab based on the word_count_threshold%d...' %word_count_threshold)
    vocabs_count = {}
    num = 0

    for label in total_labels:
        captions = Remove_redundent(label['caption'])
        for sentence in captions:
            num += 1
            for word in sentence.lower().split(' '):
                if(word not in vocabs_count): vocabs_count[word] = 0
                vocabs_count[word] += 1
    words = []
    for word in sorted(vocabs_count.keys()):
        if(vocabs_count[word] >= word_count_threshold):
            words.append(word)
    print('Filtered words from %d to %d' % (len(vocabs_count), len(words)))

    wordtoindx = {}
    wordtoindx['<pad>'] = 0
    wordtoindx['<bos>'] = 1
    wordtoindx['<eos>'] = 2
    wordtoindx['<unk>'] = 3

    indxtoword = {}
    indxtoword[0] = '<pad>'
    indxtoword[1] = '<bos>'
    indxtoword[2] = '<eos>'
    indxtoword[3] = '<unk>'

    for idx, w in enumerate(words):
        wordtoindx[w] = idx+4
        indxtoword[idx+4] = w

    vocabs_count['<pad>'] = num
    vocabs_count['<bos>'] = num
    vocabs_count['<eos>'] = num
    vocabs_count['<unk>'] = num

    bias_init_vec = np.array([1.0 * vocabs_count[ indxtoword[i] ] for i in indxtoword])
    bias_init_vec /= np.sum(bias_init_vec) #Normalize to frequencies
    bias_init_vec = np.log(bias_init_vec)
    bias_init_vec -= np.max(bias_init_vec)
    return wordtoindx, indxtoword, bias_init_vec

def Build_model(n_words, bias_init_vec=None):
    video_feat = tf.placeholder(tf.float32, [batch_size,  video_lstm_step,  dim_image])
    caption = tf.placeholder(tf.int32, [batch_size,  caption_lstm_step + 2 ])

    caption_mask = tf.placeholder(tf.float32, [batch_size, caption_lstm_step+1])

    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video_feat, dtype=tf.float32, time_major=False) #(batches, steps, inputs)

    attention_mechanism = tf.contrib.seq2seq.LuongAttention(dim_hidden, encoder_outputs)

    with tf.variable_scope('Embedding'):
        embedding_decoder = tf.Variable(tf.truncated_normal(shape=[n_words, dim_hidden], stddev=0.1), name='embedding_decoder')
        decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, caption[:, :-1])

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size= dim_hidden)

    decoder_seq_length = [caption_lstm_step+1] * batch_size

    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_emb_inp, decoder_seq_length, embedding_decoder, 0.2, time_major=False)

    projection_layer = Dense(n_words, use_bias=False )
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
    decoder_cell.zero_state( batch_size, tf.float32).clone(cell_state=encoder_state), output_layer = projection_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

    logits = outputs.rnn_output
    result = outputs.sample_id

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=caption[:,1:], logits=logits)
    train_loss = (tf.reduce_sum(cross_entropy * caption_mask) / batch_size)

    return train_loss, video_feat, caption, caption_mask, outputs.sample_id

def Build_generator(n_words, bias_init_vec=None):
    batch_size = 1

    video_feat = tf.placeholder(tf.float32, [batch_size, video_lstm_step, dim_image])

    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video_feat, dtype=tf.float32, time_major=False)

    attention_mechanism = tf.contrib.seq2seq.LuongAttention( dim_hidden, encoder_outputs)

    with tf.variable_scope('Embedding'):
        embedding_decoder = tf.Variable(tf.truncated_normal(shape=[ n_words,  dim_hidden], stddev=0.1), name='embedding_decoder')

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=dim_hidden)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([batch_size], 1), 2)

    projection_layer = Dense( n_words, use_bias=False )
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
        decoder_cell.zero_state( batch_size, tf.float32).clone(cell_state=encoder_state), output_layer = projection_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,  maximum_iterations=caption_lstm_step)

    result = outputs.sample_id

    generated_words = outputs.sample_id

    return video_feat, generated_words

def train(model_path):
    with open(os.path.join(datapath, 'training_label.json')) as f:
        train_labels = json.load(f)
    with open(os.path.join(datapath, 'testing_label.json')) as f:
        test_labels = json.load(f)
    total_labels = train_labels + test_labels
    wordtoindx, indxtoword, bias_init_vec = Bulid_Wordvocab(total_labels, word_count_threshold=word_threshold)

    np.save("./Utils/wordtoindx", wordtoindx)
    np.save('./Utils/indxtoword', indxtoword)
    np.save("./Utils/bias_init_vec", bias_init_vec)

    train_data = []
    for data in train_labels:
        vedio_ID = '%s.npy' % (data['id'])
        tmp = np.load(os.path.join(datapath, 'training_data', 'feat', vedio_ID))
        train_data.append(tmp)
    train_data = np.array(train_data)

    train_loss, tf_video, tf_caption, tf_caption_mask, tf_probs = Build_model(len(wordtoindx), bias_init_vec=bias_init_vec)
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

    optimizer = tf.train.AdamOptimizer( learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    for epoch in range(0, epochs + 1):
        for start, end in zip(range(0, len(train_data), batch_size),range(batch_size, len(train_data), batch_size)):
            start_time = time.time()

            current_feats = train_data[start:end]
            current_video_masks = np.zeros((batch_size, video_lstm_step))
            current_captions = []

            for ind in range(len(current_feats)):
                current_video_masks[ind][:len(current_feats[ind])] = 1
                current_captions.append(random.choice(train_labels[start + ind]['caption']))

            current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
            current_captions = Remove_redundent(current_captions)
            current_captions = list(current_captions)

            current_captions_src = []

            for idx, each_word in enumerate(current_captions):
                word = each_word.lower().split(' ')
                if len(word) < caption_lstm_step + 1:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(caption_lstm_step):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_indx = []
            for sentence in current_captions:
                current_word_ind = []
                for word in sentence.lower().split(' '):
                    if word in wordtoindx:
                        current_word_ind.append(wordtoindx[word])
                    else:
                        current_word_ind.append(wordtoindx['<unk>'])
                current_caption_indx.append(current_word_ind)

            current_caption_matrix_src = sequence.pad_sequences(current_caption_indx, padding='post', maxlen=caption_lstm_step+1)
            current_caption_matrix_src = np.hstack([current_caption_matrix_src, np.zeros([len(current_caption_matrix_src), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix_src.shape[0], current_caption_matrix_src.shape[1]-1) )

            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix_src)))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            update_step, loss_val= sess.run([train_op, train_loss], feed_dict={
                tf_video: current_feats,
                tf_caption: current_caption_matrix_src,
                tf_caption_mask: current_caption_masks
                })
            print('index: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))

        if np.mod(epoch, 10) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=epoch)

def test(model_path, testing_path, outputfile):
    test_videos = []
    with open(os.path.join(testing_path, 'testing_id.txt')) as f:
        for line in f:
            test_videos.append(line.strip())
    indxtoword = pd.Series(np.load('./Utils/indxtoword.npy').tolist())
    bias_init_vec = np.load('./Utils/bias_init_vec.npy')

    video, words = Build_generator(len(indxtoword))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open(outputfile, 'w')
    for idx, video_feat_path in enumerate(test_videos):

        video_feat = np.load(os.path.join(testing_path, video_feat_path + '.npy'))[None,...]

        feed_dict={
            video: video_feat
        }

        probs_val = sess.run(words, feed_dict=feed_dict)

        generated_words = indxtoword[list(probs_val[0])]
        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]
        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        video_name = video_feat_path

        test_output_txt_fd.write("%s,%s\n" % (video_name, generated_sentence))

def main():
    global training_feat_path
    global datapath
    global word_threshold

    if not datapath:
        print("Please provide datapath")
        exit()

    training_feat_path = os.path.join(datapath, 'training_data', 'feat')

    #train(model_path)
    test(model_path, sys.argv[1], sys.argv[2])

if __name__ == "__main__":
	main()
