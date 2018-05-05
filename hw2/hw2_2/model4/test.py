import sys

import tensorflow as tf

import model4.model
import model4.config as cfg

import util.util as util
import util.data_util as data_util

def main():
    data, data_len = data_util.read_test_data(sys.argv[1], 'model4/int2str.pkl', 'model4/str2int.pkl', cfg.INPUT_MAX_LEN)
    int2str, str2int = data_util._load_dict('model4/int2str.pkl', 'model4/str2int.pkl')

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        model_eval = model4.model.EvalModel(
            cfg.INPUT_MAX_LEN,
            cfg.DICT_SIZE,
            cfg.RNN_SIZE,
            cfg.RNN_LAYER,
            cfg.BATCH_SIZE
        )

    test_graph = tf.Graph()
    with test_graph.as_default():
        model_test = model4.model.TestModel(
            cfg.INPUT_MAX_LEN, 
            cfg.DICT_SIZE, 
            cfg.RNN_SIZE, 
            cfg.RNN_LAYER,
            cfg.BATCH_SIZE,
            cfg.BEAM_WIDTH
        )
    
    eval_sess = tf.Session(graph=eval_graph)
    test_sess = tf.Session(graph=test_graph)

    model_eval.load(eval_sess, name=cfg.MODEL_SAVE_PATH)
    model_test.load(test_sess, name=cfg.MODEL_SAVE_PATH)

    ans = []
    ans2 = []
    batches = len(data)//cfg.BATCH_SIZE
    for batch in range(batches):
        greedy = model_eval.greedy(
            eval_sess, 
            data[batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE], 
            data_len[batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE]
        )
        result = model_test.predict(
            test_sess, 
            data[batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE], 
            data_len[batch*cfg.BATCH_SIZE:(batch+1)*cfg.BATCH_SIZE]
        )
        for i in range(len(greedy)):
            inserted = False
            if len(util.adjust_int_output(greedy[i]))>0:
                ans2.append(util.int_to_str(int2str, util.adjust_int_output(greedy[i])))
                inserted = True
            if not inserted:
                ans2.append('我不知道')
        for i in range(len(result)):
            inserted = False
            for j in range(cfg.BEAM_WIDTH):
                if len(util.adjust_int_output(result[i][j]))>0:
                    ans.append(util.int_to_str(int2str, util.adjust_int_output(result[i][j])))
                    inserted = True
                    break
            if not inserted:
                ans.append('我不知道')
    remain = len(data) - batches*cfg.BATCH_SIZE
    if remain>0:
        greedy = model_eval.greedy(
            eval_sess,
            data[-remain:] + [[0]*cfg.INPUT_MAX_LEN] * (cfg.BATCH_SIZE-remain),
            data_len[-remain:] + [1] * (cfg.BATCH_SIZE-remain)
        )
        result = model_test.predict(
            test_sess,
            data[-remain:] + [[0]*cfg.INPUT_MAX_LEN] * (cfg.BATCH_SIZE-remain),
            data_len[-remain:] + [1] * (cfg.BATCH_SIZE-remain)
        )
        for i in range(remain):
            inserted = False
            if len(util.adjust_int_output(greedy[i]))>0:
                ans2.append(util.int_to_str(int2str, util.adjust_int_output(greedy[i])))
                inserted = True
            if not inserted:
                ans2.append('我不知道')
        for i in range(remain):
            inserted = False
            for j in range(cfg.BEAM_WIDTH):
                if len(util.adjust_int_output(result[i][j]))>0:
                    ans.append(util.int_to_str(int2str, util.adjust_int_output(result[i][j])))
                    inserted = True
                    break
            if not inserted:
                ans.append('我不知道')
    print(ans2)
    with open(sys.argv[2], 'w') as w:
        for i in range(len(ans)):
            w.write(ans2[i]+'\n')

if __name__=='__main__':
    main()