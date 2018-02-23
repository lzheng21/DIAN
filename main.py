import random
from load_data import *
from models.BPR import *
import tensorflow as tf
#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 32
BATCH_SIZE = 1024
lambda_u = 0.02
lambda_v = 0.02
K = 3
N_EPOCH = 50000
LR = 0.001

DIR = 'data/ml-1m/'


data_generator = Data(batch_size=BATCH_SIZE, train_file=DIR+'train_users.dat', test_file=DIR+'test_users.dat')
model = BPR(data_generator, EMB_DIM, BATCH_SIZE, lambda_u=lambda_u, lambda_v=lambda_v)
sess = model.train(N_EPOCH, lr=LR)
model.predict()
