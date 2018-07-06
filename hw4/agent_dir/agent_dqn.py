from agent_dir.agent import Agent
from collections import deque
import tensorflow as tf
#import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(10)

FINAL_EXPLORATION = 0.025
TARGET_UPDATE = 1000
ONLINE_UPDATE = 4

MEMORY_SIZE = 10000
EXPLORATION = 1000000

START_EXPLORATION = 1.
TRAIN_START = 10000
LEARNING_RATE = 0.00015
DISCOUNT = 0.99


class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        tf.reset_default_graph()
        self.num_action = 3
        self.minibatch = 32
        self.esp = 1
        self.model_path = "./model/Breakout_ddqn.ckpt"
        self.replay_memory = deque()


        self.input = tf.placeholder("float", [None, 84, 84, 4])

        self.f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2 = tf.get_variable("w2", shape=[512, self.num_action], initializer=tf.contrib.layers.xavier_initializer())

        self.py_x = self.build_model(self.input, self.f1, self.f2, self.f3 , self.w1, self.w2)


        self.f1_r = tf.get_variable("f1_r", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2_r = tf.get_variable("f2_r", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3_r = tf.get_variable("f3_r", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1_r = tf.get_variable("w1_r", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2_r = tf.get_variable("w2_r", shape=[512, self.num_action], initializer=tf.contrib.layers.xavier_initializer())

        self.py_x_r =self.build_model(self.input, self.f1_r, self.f2_r,self.f3_r, self.w1_r, self.w2_r)

        self.rlist=[]
        self.recent_rlist=[]
        self.avg_r_thirty_list=[]
        self.avg_r_thirty_list_out=[]

        self.episode = 0
        self.avg_r_thirty = 0

        self.a= tf.placeholder(tf.int64, [None])
        self.y = tf.placeholder(tf.float32, [None])
        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        a_one_hot = tf.one_hot(self.a, self.num_action, 1.0, 0.0)
        self.q_value = tf.reduce_sum(tf.multiply(self.py_x, a_one_hot), reduction_indices=1)

        error = tf.abs(self.q_target - self.q_value)

        diff = tf.where(tf.abs(error) < 1.0, 0.5 * tf.square(error), tf.abs(error) - 0.5)

        self.loss = tf.reduce_mean(tf.reduce_sum(diff))

        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=0, epsilon= 1e-8, decay=0.99)
        self.train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=None)

        cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
        self.sess = tf.Session(config=cfg)

        if args.test_dqn:
            self.saver.restore(self.sess, save_path = self.model_path)
            print('loading trained model')

    def build_model(self, input1, f1, f2, f3, w1, w2):

        c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1],data_format="NHWC", padding = "VALID"))
        c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1],data_format="NHWC", padding="VALID"))
        c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],data_format="NHWC", padding="VALID"))

        l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])

        l2 = tf.maximum((tf.matmul(l1, w1)), 0.01 * (tf.matmul(l1, w1)))
        pyx = tf.matmul(l2, w2)

        return pyx

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        pass


    def train(self):
        """
        Implement your training algorithm here
        """

        f = open('reward_dqn_1.txt', 'w')
        frame = 0
        self.rlist.append(0)
        self.recent_rlist.append(0)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.w1_r.assign(self.w1))
        self.sess.run(self.w2_r.assign(self.w2))
        self.sess.run(self.f1_r.assign(self.f1))
        self.sess.run(self.f2_r.assign(self.f2))
        self.sess.run(self.f3_r.assign(self.f3))

        while np.mean(self.recent_rlist) < 500 :
            self.episode += 1

            if len(self.recent_rlist) > 100:
                del self.recent_rlist[0]

            rall = 0
            r_unclip_all = 0
            d = False
            ter = False
            count = 0
            s = self.env.reset()
            avg_max_Q = 0
            avg_loss = 0

            while not d :

                frame +=1
                count+=1

                if self.esp > FINAL_EXPLORATION and frame > TRAIN_START:
                    self.esp -=  (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                state = np.reshape(s, (1, 84, 84, 4))
                Q = self.sess.run(self.py_x, feed_dict = {self.input : state})

                avg_max_Q += np.max(Q)

                if self.esp > np.random.rand(1):
                    action = np.random.randint(self.num_action)
                else:
                    action = np.argmax(Q)

                if action == 0:
                    real_a = 1
                elif action == 1:
                    real_a = 2
                else:
                    real_a = 3

                s1, r, d, l = self.env.step(real_a)

                r_unclip = r

                r = np.sign(r)

                ter = d

                self.replay_memory.append((np.copy(s), np.copy(s1), action ,r, ter))

                s = s1

                if len(self.replay_memory) > MEMORY_SIZE:
                    self.replay_memory.popleft()

                rall += r
                r_unclip_all += r_unclip

                if frame > TRAIN_START and frame % ONLINE_UPDATE == 0:
                    s_stack = deque()
                    a_stack = deque()
                    r_stack = deque()
                    s1_stack = deque()
                    d_stack = deque()
                    y_stack = deque()

                    sample = random.sample(self.replay_memory, self.minibatch)

                    for _s , s_r, a_r, r_r, d_r in sample:
                        s_stack.append(_s)
                        a_stack.append(a_r)
                        r_stack.append(r_r)
                        s1_stack.append(s_r)
                        d_stack.append(d_r)

                    d_stack = np.array(d_stack)

                    Q1 = self.sess.run( self.py_x_r, feed_dict={self.input: np.array(s1_stack)})

                    Q_online = self.sess.run(self.py_x, feed_dict={self.input: np.array(s1_stack)})

                    max_act4next = np.argmax(Q_online, axis=1)

                    Q1_d = []
                    Q1_d_count = 0
                    for each_Q1 in Q1:
                        Q1_d.append(each_Q1[max_act4next[Q1_d_count]])

                    expected_state_action_values = r_stack + (1 - d_stack) * DISCOUNT * np.array(Q1_d)

                    self.sess.run(self.train_op,feed_dict={self.input: np.array(s_stack),self.a:a_stack, self.q_target: expected_state_action_values})

                    if frame % TARGET_UPDATE == 0 :
                        self.sess.run(self.w1_r.assign(self.w1))
                        self.sess.run(self.w2_r.assign(self.w2))
                        self.sess.run(self.f1_r.assign(self.f1))
                        self.sess.run(self.f2_r.assign(self.f2))
                        self.sess.run(self.f3_r.assign(self.f3))

            if (frame - TRAIN_START) > 50000 == 0:

                save_path = self.saver.save(self.sess, self.model_path)
                print("Model(episode :",self.episode, ") saved in file: ", save_path )

            self.recent_rlist.append(rall)
            self.rlist.append(rall)
            self.avg_r_thirty_list.append(r_unclip_all)

            if self.episode % 10 == 0:
                print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | Avg_Max_Q:{4:2.5f} | "
                    "Recent reward:{5:.5f}  ".format(self.episode, frame, count, rall, avg_max_Q/float(count),np.mean(self.recent_rlist)))

                s = "%d\t%lf\n" % (self.episode, np.mean(self.recent_rlist))
                f.write(s)

            if self.episode % 30 == 0:
                self.avg_r_thirty = np.mean(self.avg_r_thirty_list)
                self.avg_r_thirty_list=[]

            if self.episode > 30:
                self.avg_r_thirty_list_out.append(self.avg_r_thirty)
                np.save("Loss_curve_ddqn.npy", np.array(self.avg_r_thirty_list_out))


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        state = np.reshape(observation, (1, 84, 84, 4))
        Q = self.sess.run(self.py_x, feed_dict = {self.input : state})

        self.esp = 0.01

        if self.esp > np.random.rand(1):
            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(Q)

        if action == 0:
            real_a = 1
        elif action == 1:
            real_a = 2
        else:
            real_a = 3

        return real_a
