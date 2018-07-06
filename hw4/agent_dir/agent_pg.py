from agent_dir.agent import Agent
import scipy
import numpy as np
import tensorflow as tf

# def prepro(o,image_size=[80,80]):
#     """
#     Call this function to preprocess RGB image to grayscale image if necessary
#     This preprocessing code is from
#         https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
#     Input: 
#     RGB image: np.array
#         RGB screen of game, shape: (210, 160, 3)
#     Default return: np.array 
#         Grayscale image, shape: (80, 80, 1)
    
#     """
#     y = o.astype(np.uint8)
#     resized = scipy.misc.imresize(y, image_size)
#     return np.expand_dims(resized.astype(np.float32),axis=2)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        self.n_obs = 80 * 80           # dimensionality of observations
        self.h = 256                   # number of hidden layer neurons
        self.n_actions = 3             # number of available actions
        self.learning_rate = 1e-3
        self.gamma = .99               # discount factor for reward
        self.decay = 0.99              # decay rate for RMSProp gradients
        self.save_path="model/pong.ckpt"

        # freeze random seed
        self.seed = 11037
        self.env.seed(self.seed)

        self.last_observation = None

        self.tf_model = {}
        with tf.variable_scope('layer_one',reuse=False):
            self.xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
            self.tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.h], initializer=self.xavier_l1)
        with tf.variable_scope('layer_two',reuse=False):
            self.xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.h), dtype=tf.float32)
            self.tf_model['W2'] = tf.get_variable("W2", [self.h, self.n_actions], initializer=self.xavier_l2)

        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs],name="tf_x")
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions],name="tf_y")
        self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")

        # tf reward processing (need tf_discounted_epr for policy gradient wizardry)
        self.tf_discounted_epr = self.tf_discount_rewards(self.tf_epr)
        self.tf_mean, self.tf_variance= tf.nn.moments(self.tf_discounted_epr, [0], shift=None, name="reward_moments")
        self.tf_discounted_epr -= self.tf_mean
        self.tf_discounted_epr /= tf.sqrt(self.tf_variance + 1e-6)

        # tf optimizer op
        self.tf_aprob = self.tf_policy_forward(self.tf_x)
        self.loss = tf.nn.l2_loss(self.tf_y-self.tf_aprob)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay)
        self.tf_grads = self.optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(), grad_loss=self.tf_discounted_epr)
        self.capped_tf_grads = [(tf.clip_by_value(grad, -0.3, 0.3), var) for grad, var in self.tf_grads]
        self.train_op = self.optimizer.apply_gradients(self.capped_tf_grads)

        # tf graph initialization
        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.saver = None

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.saver = tf.train.Saver(tf.all_variables())
            self.saver.restore(self.sess, self.save_path)

    def tf_discount_rewards(self, tf_r): 
        discount_f = lambda a, v: a* self.gamma + v
        tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
        tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
        return tf_discounted_r
    
    def tf_policy_forward(self, x): 
        h = tf.matmul(x, self.tf_model['W1'])
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self.tf_model['W2'])
        p = tf.nn.softmax(logp)
        return p

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.last_observation = None


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        self.saver = tf.train.Saver()

        observation = self.env.reset()
        prev_x = None
        xs, rs, ys = [],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        while True:
            # env.render()

            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.n_obs)
            prev_x = cur_x

            feed = {self.tf_x: np.reshape(x, (1,-1))}
            aprob = self.sess.run(self.tf_aprob,feed) ; aprob = aprob[0,:]
            action = np.random.choice(self.n_actions, p=aprob)
            label = np.zeros_like(aprob) ; label[action] = 1

            observation, reward, done, info = self.env.step(action+1)
            reward_sum += reward

            xs.append(x) ; ys.append(label) ; rs.append(reward)
            if done:

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01 

                feed = {self.tf_x: np.vstack(xs), self.tf_epr: np.vstack(rs), self.tf_y: np.vstack(ys)}
                _ = self.sess.run(self.train_op,feed)
                # print progress console
                print('ep %d: reward: %d' % (episode_number, reward_sum))

                xs,rs,ys = [],[],[] 
                episode_number += 1 
                observation = self.env.reset()
                reward_sum = 0
                if episode_number % 1 == 0:
                    self.saver.save(self.sess, self.save_path, global_step=episode_number)


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        cur_x = prepro(observation)
        x = cur_x - self.last_observation if self.last_observation is not None else np.zeros(self.n_obs)
        self.last_observation = cur_x
        feed = {self.tf_x: np.reshape(x, (1,-1))}
        aprob = self.sess.run(self.tf_aprob,feed) ; aprob = aprob[0,:]
        action = np.random.choice(self.n_actions, p=aprob)
        action = action + 1
        return action

