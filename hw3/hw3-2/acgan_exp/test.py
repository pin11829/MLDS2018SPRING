import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt

import acgan_exp.model as Model
import acgan_exp.config as cfg
import util.util as util

def main(tags_path, output_path):
    test_graph = tf.Graph()
    with test_graph.as_default():
        tf.set_random_seed(299) # 3
        test_model = Model.TestModel(
            cfg.BATCH_SIZE, 
            cfg.NOISE_DIM, 
            cfg.COND1_DIM, 
            cfg.COND2_DIM,
            cfg.G_BLOCKS
        )
    test_sess = tf.Session(graph=test_graph)
    test_model.load(test_sess, os.path.join(cfg.PATH_PREFIX, 'ckpt'))
    
    with open(tags_path, 'r') as f:
        data = f.read().strip().split('\n')
    cond1 = []
    cond2 = []
    for i in range(len(data)):
        for j in range(cfg.COND1_DIM):
            if cfg.COND1[j] in data[i]:
                cond1.append(j)
                break
        for j in range(cfg.COND2_DIM):
            if cfg.COND2[j] in data[i]:
                cond2.append(j)
                break
    cond1 = cond1 + [0]*(cfg.BATCH_SIZE-len(cond1))
    cond2 = cond2 + [0]*(cfg.BATCH_SIZE-len(cond2))
    images = test_model.sample_images(test_sess, cond1, cond2) ### TODO: modify batch

    #
    r, c = 5, 5
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = (images[:25]+1)/2.0
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :, :, :])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(output_path)
    plt.close()

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])