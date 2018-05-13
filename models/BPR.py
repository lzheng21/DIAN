import tensorflow as tf
import utils as ut
import numpy as np
from losses import *
import multiprocessing

class BPR(object):
    def __init__(self, data, emb_dim, batch_size, lambda_u=0.001, lambda_v=0.001):
        self.data = data
        self.n_users, self.n_items = self.data.get_num_users_items()
        self.emb_dim = emb_dim
        self.batch_size = batch_size

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.neg_items = tf.placeholder(tf.int32, shape=(self.batch_size,))

        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')


        self.u_embeddings = tf.gather(self.user_embeddings, self.users)

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_embeddings, transpose_a=False,
                                    transpose_b=True)

        self.pos_item_embeddings = tf.gather(self.item_embeddings, self.pos_items)
        self.neg_item_embeddings = tf.gather(self.item_embeddings, self.neg_items)

        self.loss = bpr_loss(self.u_embeddings, self.pos_item_embeddings, self.neg_item_embeddings, lambda_u, lambda_v)

        self.params = [self.user_embeddings, self.item_embeddings]

    def train(self, n_epoch, lr, optimizer):
        assert optimizer in {'Adam', 'SGD'}

        if optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(lr)
        if optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(lr)

        self.updates = self.opt.minimize(self.loss, var_list=self.params)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):
            users, pos_items, neg_items = self.data.sample_pairs(self.batch_size)
            _, loss = self.sess.run([self.updates, self.loss],
                                   feed_dict={self.users: users,
                                              self.pos_items: pos_items,
                                              self.neg_items: neg_items})

            ret = self.predict(mode=1)
            p_3, ndcg_3, MAP = ret[0], ret[1], ret[2]
            print('\rEpoch %d training loss %f val Precision_3 %f val NDCG_3 %f val MAP %f' % \
                  (epoch, loss, p_3, ndcg_3, MAP), end='')



    def predict(self,mode=0):
        result = np.array([0.] * 3)
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cores)

        # all users needed to test
        test_users = list(self.data.test_set.keys()) if mode == 0 else list(self.data.val_set.keys())

        test_user_num = len(test_users)
        index = 0
        while True:
            if index >= test_user_num:
                break
            user_batch = test_users[index:index + self.batch_size]
            index += self.batch_size
            FLAG = False
            if len(user_batch) < self.batch_size:
                user_batch += [0] * (self.batch_size - len(user_batch))
                user_batch_len = len(user_batch)
                FLAG = True
            user_batch_rating = self.sess.run(self.all_ratings, {self.users: user_batch})
            user_batch_rating_uid = zip(user_batch_rating, user_batch, [self.data] * self.batch_size, [mode]*self.batch_size)
            batch_result = pool.map(ut.test_one_user, user_batch_rating_uid)
            if FLAG == True:
                batch_result = batch_result[:user_batch_len]
            for re in batch_result:
                result += re

        pool.close()
        ret = result / test_user_num
        ret = list(ret)
        if mode == 1: return ret
        p_3, ndcg_3, MAP = ret[0], ret[1], ret[2]

        print('\nPrecision_3 %f NDCG_3 %f MAP %f' % (p_3, ndcg_3, MAP), end='')












