import tensorflow as tf
import utils as ut
import numpy as np
from losses import *
class BPR(object):
    def __init__(self, data, emb_dim, lr, batch_size, lambda_u, lambda_v):
        self.name = 'BPR'
        self.n_users, self.n_items = data.get_num_users_items()
        self.emb_dim = emb_dim

        self.batch_size = batch_size

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.neg_items = tf.placeholder(tf.int32, shape=(self.batch_size,))


        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], dtype=tf.float32),
            name='item_embeddings')


        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_embeddings, transpose_a=False,
                                    transpose_b=True)

        self.pos_item_embeddings = tf.gather(self.item_embeddings, self.pos_items)
        self.neg_item_embeddings = tf.gather(self.item_embeddings, self.neg_items)

        self.loss = bpr_loss(self.user_embeddings, self.pos_item_embeddings, self.neg_item_embeddings, lambda_u, lambda_v)

        self.params = [self.user_embeddings, self.item_embeddings]
        self.opt = tf.train.AdamOptimizer(lr)
        self.updates = self.opt.minimize(self.loss, var_list=self.params)










