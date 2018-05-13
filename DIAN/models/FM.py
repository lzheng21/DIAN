import multiprocessing

import numpy as np

from DIAN import utils as ut
from DIAN.losses import *


class FM(object):
    def __init__(self, data, user_emb_dim, item_emb_dim, factor_dim, batch_size, weights,lambda_u=0.01, lambda_v=0.01):
        '''
        :param data: A Data object
        :param user_emb_dim: int, dimension of user vectors
        :param item_emb_dim: int, dimension of item vectors
        :param factor_dim: int, dimension of the factorized matrix
        :param batch_size: int, batch size
        :param weights: dictionary: {0:weight, 1:weight}, weight for each class 
        :param lambda_u: int, regularization weight
        :param lambda_v: int, regularization weight
        '''

        self.data = data
        self.n_users, self.n_items = self.data.get_num_users_items()
        self.user_emb_dim = user_emb_dim
        self.item_emb_dim = item_emb_dim
        self.k = factor_dim
        self.batch_size = batch_size
        self.weights = weights

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.items = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.labels = tf.placeholder(tf.int32, shape=(self.batch_size,))


        #variables
        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.user_emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.item_emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')

        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.items)


        self.w0 = tf.Variable(tf.zeros([1]))
        self.W = tf.Variable(tf.zeros([user_emb_dim+item_emb_dim]))
        # interaction factors, randomly initialized
        self.V = tf.Variable(tf.random_normal([self.k, user_emb_dim+item_emb_dim], stddev=0.01))
        # estimate of y, initialized to 0.
        self.X = tf.concat([self.u_embeddings,self.i_embeddings],axis=[1])
        #build
        self.linear_terms = tf.add(self.w0,
                              tf.reduce_sum(
                                  tf.multiply(self.W, self.X), 1, keep_dims=True))

        self.interactions = (tf.multiply(0.5,
                                    tf.reduce_sum(
                                        tf.subtract(
                                            tf.pow(tf.matmul(self.X, tf.transpose(self.V)), 2),
                                            tf.matmul(tf.pow(self.X, 2), tf.transpose(tf.pow(self.V, 2)))),
                                        1, keep_dims=True)))
        self.y_hat = tf.add(self.linear_terms, self.interactions)

        self.loss = tf.losses.mean_squared_error(self.labels,self.y_hat) + \
                    lambda_u * tf.nn.l2_loss(self.u_embeddings) + lambda_v * tf.nn.l2_loss(self.i_embeddings)
        self.params = [self.user_embeddings, self.item_embeddings, self.w0, self.W, self.V]

        FIRST = True
        for uid in tf.unstack(self.users):
            u_embedding = tf.gather(self.user_embeddings,uid)
            u_embeddings = tf.concat([u_embedding]*self.n_items,axis=[1])
            X = tf.concat([u_embeddings,self.item_embeddings],axis=[1])
            linear_terms = tf.add(self.w0,
                                       tf.reduce_sum(
                                           tf.multiply(self.W, X), 1, keep_dims=True))

            interactions = (tf.multiply(0.5,
                                             tf.reduce_sum(
                                                 tf.subtract(
                                                     tf.pow(tf.matmul(X, tf.transpose(self.V)), 2),
                                                     tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(self.V, 2)))),
                                                 1, keep_dims=True)))
            self.ratings_one_user = tf.add(linear_terms, interactions)
            if FIRST:
                self.all_ratings = self.ratings_one_user
                FIRST = False
            self.all_ratings = tf.stack([self.all_ratings,self.ratings_one_user])





    def train(self, n_epoch, lr, optimizer):
        assert optimizer in {'Adam', 'SGD'}

        if optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(lr)
        if optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(lr)

        self.opt = tf.train.AdamOptimizer(lr)
        self.updates = self.opt.minimize(self.loss, var_list=self.params)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):
            users, items, labels = self.data.sample(batch_size=self.batch_size)
            #add weight
            labels *= [l*self.weights[1] for l in labels]
            _, loss = self.sess.run([self.updates, self.loss],
                                   feed_dict={self.users: users,
                                              self.items: items,
                                              self.labels: labels})
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
            user_batch_rating_uid = zip(user_batch_rating, user_batch, [self.data] * self.batch_size)
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
        print('Precision_3 %f NDCG_3 %f MAP %f' % (p_3, ndcg_3, MAP), end='')










