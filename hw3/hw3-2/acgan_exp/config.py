HEIGHT = WIDTH = 96
G_BLOCKS = 16
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999
ITERATION = int(10e4)
G_ITERATION = 1
D_ITERATION = 1
SAVE_PERIOD = 100
SAMPLE_PERIOD = 50
DISPLAY_PERIOD = 5

NOISE_DIM = 128
COND1 = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair'] # 12
COND2 =  ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes'] # 11
COND1_DIM = len(COND1)
COND2_DIM = len(COND2)

G_LAMBDA = 25
D_LAMBDA = 25

LOAD_MODEL = True

PATH_PREFIX = 'hw3-2/acgan_exp'
