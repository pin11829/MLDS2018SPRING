import tensorflow as tf
import numpy as np

import time
import os

import model4.config as cfg
import model4.model

import util.data_util as data_util
import util.util as util
from util.util import adjust_int_output

if __name__=='__main__':
    data_input, data_output, data_input_len, data_output_len, data_output_shift, int2str, str2int = data_util.read_data(
        filename='data/clr_conversation.txt',
        build_dict=cfg.BUILD_DICT,
        filename_int2str= os.path.join(cfg.MODEL_SAVE_PATH, 'int2str.pkl'),
        filename_str2int= os.path.join(cfg.MODEL_SAVE_PATH, 'str2int.pkl'),
        valid=cfg.VALID_MODEL,
        reshuffle=cfg.RESHUFFLE,
        shuffle_file= os.path.join(cfg.MODEL_SAVE_PATH, 'shuffle.npy'),
        input_max_len=cfg.INPUT_MAX_LEN,
        min_count=cfg.WORD_MIN_COUNT
    )

    train_graph = tf.Graph()
    eval_graph = tf.Graph()
    test_graph = tf.Graph()

    with train_graph.as_default():
        model = model4.model.TrainModel(
            data_input['train'],
            data_output['train'],
            data_input_len['train'],
            data_output_len['train'],
            data_output_shift['train'],
            cfg.INPUT_MAX_LEN,
            cfg.DICT_SIZE,
            cfg.RNN_SIZE,
            cfg.RNN_LAYER,
            cfg.SCHEDULE_RATE,
            cfg.LEARNING_RATE,
            cfg.LEARNING_RATE_DECAY_STEPS,
            cfg.LEARNING_RATE_DECAY_RATE,
            cfg.MAX_GRADIENT_NORM,
            cfg.LSTM_KEEPPROB,
            cfg.BATCH_SIZE
        )
    with eval_graph.as_default():
        model_eval = model4.model.EvalModel(
            cfg.INPUT_MAX_LEN, 
            cfg.DICT_SIZE, 
            cfg.RNN_SIZE, 
            cfg.RNN_LAYER, 
            cfg.BATCH_SIZE
        )
    with test_graph.as_default():
        model_test = model4.model.TestModel(
            cfg.INPUT_MAX_LEN,
            cfg.DICT_SIZE,
            cfg.RNN_SIZE,
            cfg.RNN_LAYER,
            cfg.BATCH_SIZE,
            cfg.BEAM_WIDTH
        )
    
    gpu_options1 = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_options2 = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    gpu_options3 = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    

    '''config1 = tf.ConfigProto(gpu_options=gpu_options1)
    config2 = tf.ConfigProto(gpu_options=gpu_options2)
    config3 = tf.ConfigProto(gpu_options=gpu_options3)'''

    config1 = tf.ConfigProto()
    #config1.gpu_options.allow_growth=True 
    config2 = tf.ConfigProto()
    #config2.gpu_options.allow_growth=True 
    config3 = tf.ConfigProto()
    #config3.gpu_options.allow_growth=True 

    train_sess = tf.Session(graph=train_graph, config=config1)
    eval_sess = tf.Session(graph=eval_graph, config=config2)
    test_sess = tf.Session(graph=test_graph, config=config3)

    if cfg.LOAD_MODEL:
        model.load(train_sess, name=cfg.MODEL_SAVE_PATH)
    else:
        model.init(train_sess)

    model.prepare_dataset(
        train_sess, 
        data_input['train'], 
        data_output['train'], 
        data_input_len['train'], 
        data_output_len['train'],
        data_output_shift['train']
    )

    for epoch in range(cfg.EPOCH):

        batches = len(data_input['train'])//cfg.BATCH_SIZE

        start_time = time.time()
        print('Current lr=', model.get_learning_rate(train_sess))
        print('Current step=', model.get_global_step(train_sess))

        for batch in range(batches): #
            train_loss, gn = model.train(train_sess)
            if (batch+1)%cfg.PRINT_ITERATION==0:
                print('Epoch: {0:2d}/{1:2d}, Batch: {2:5d}/{3:5d}, Loss={4:5.4f}, Time={5:3.2f}, |g|={6:3.4f}'.format(
                    epoch+1,
                    cfg.EPOCH,
                    batch+1,
                    batches,
                    train_loss,
                    time.time()-start_time,
                    gn)
                )
                start_time = time.time()
        model.save(train_sess, cfg.MODEL_SAVE_PATH)

        if cfg.VALID_MODEL:
            model_eval.load(eval_sess, name=cfg.MODEL_SAVE_PATH)
            print('Training data examples:')
            train_loss, train_output, greedy_output = model_eval.eval(
                eval_sess,
                data_input['train'][:cfg.BATCH_SIZE],
                data_input_len['train'][:cfg.BATCH_SIZE],
                data_output['train'][:cfg.BATCH_SIZE],
                data_output_len['train'][:cfg.BATCH_SIZE],
                data_output_shift['train'][:cfg.BATCH_SIZE]
            )
            for i in range(min(cfg.BATCH_SIZE, 10)):
                a = util.int_to_str(int2str, adjust_int_output(data_input['train'][i]))
                print('Input :', a)
                b = util.int_to_str(int2str, adjust_int_output(data_output['train'][i]))
                print('Output:', b)
                c = util.int_to_str(int2str, adjust_int_output(train_output[i]))
                print(c)
                d = util.int_to_str(int2str, adjust_int_output(greedy_output[i]))
                print(d)
                print()
            
            print('Validating...')
            batches = len(data_input['valid'])//cfg.BATCH_SIZE
            vl = []
            for batch in range(100): #
                valid_loss, train_output, greedy_output = model_eval.eval(
                    eval_sess,
                    data_input['valid'][batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE],
                    data_input_len['valid'][batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE],
                    data_output['valid'][batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE],
                    data_output_len['valid'][batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE],
                    data_output_shift['valid'][batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE]
                )
                vl.append(valid_loss) 
            print(vl)
            print('Epoch: {0:2d}/{1:2d}, Validation loss={2:5.4f}'.format(epoch+1, cfg.EPOCH, sum(vl)/len(vl)))
            for i in range(min(cfg.BATCH_SIZE, 10)):
                a = util.int_to_str(int2str, adjust_int_output(data_input['valid'][99*cfg.BATCH_SIZE:100*cfg.BATCH_SIZE][i]))
                print('Input :', a)
                b = util.int_to_str(int2str, adjust_int_output(data_output['valid'][99*cfg.BATCH_SIZE:100*cfg.BATCH_SIZE][i]))
                print('Output:', b)
                c = util.int_to_str(int2str, adjust_int_output(train_output[i]))
                print(c)
                d = util.int_to_str(int2str, adjust_int_output(greedy_output[i]))
                print(d)
                print()

            # Beam
            print('Beam Searching...')
            model_test.load(test_sess, name=cfg.MODEL_SAVE_PATH)
            result = model_test.predict(
                test_sess,
                data_input['valid'][(batches-2)*cfg.BATCH_SIZE:(batches-1)*cfg.BATCH_SIZE],
                data_input_len['valid'][(batches-2)*cfg.BATCH_SIZE:(batches-1)*cfg.BATCH_SIZE],
            )
            for i in range(min(len(result), 20)):
                a = util.int_to_str(int2str, adjust_int_output(data_input['valid'][(batches-2)*cfg.BATCH_SIZE:(batches-1)*cfg.BATCH_SIZE][i]))
                print('Input :', a)
                b = util.int_to_str(int2str, adjust_int_output(data_output['valid'][(batches-2)*cfg.BATCH_SIZE:(batches-1)*cfg.BATCH_SIZE][i]))
                print('Output:', b)
                for j in range(cfg.BEAM_WIDTH):
                    c = util.int_to_str(int2str, adjust_int_output(result[i][j]))
                    print(c)
                print() 
