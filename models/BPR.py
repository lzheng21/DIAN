import tensorflow as tf
import utils as ut
import numpy as np

class BPR(object):
    def __init__(self, n_users, n_items, emb_dim, lr, batch_size, decay):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        self.batch_size = batch_size
        self.negative_size = 1
        self.decay = decay

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.items = tf.placeholder(tf.int32, shape=(self.batch_size * (1 + self.negative_size),))

        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], dtype=tf.float32),
            name='item_embeddings')


        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_embeddings, transpose_a=False,
                                    transpose_b=True)

        self.h_user = tf.gather(self.user_embeddings, self.users)
        self.h_item = tf.gather(self.item_embeddings, self.items)
        self.loss = self.create_loss(self.h_user, self.h_item)

        self.params = [self.user_embeddings, self.item_embeddings]
        self.opt = tf.train.AdamOptimizer(lr)
        self.updates = self.opt.minimize(self.loss, var_list=self.params)

    def create_loss(self, users, items):
        losses = []

        for u, i in zip(tf.split(users, num_or_size_splits=self.batch_size, axis=0),
                        tf.split(items, num_or_size_splits=self.batch_size, axis=0)):
            pos = tf.gather(i,0)
            neg = tf.gather(i,1)
            pos_score = tf.reduce_sum(tf.multiply(u, pos))
            neg_score = tf.reduce_sum(tf.multiply(u, neg))

            # Loss function using L2 Regularization
            regularizer = tf.nn.l2_loss(u) + tf.nn.l2_loss(pos) + tf.nn.l2_loss(neg)
            maxi = tf.log(tf.nn.sigmoid(pos_score - neg_score)) - self.decay * regularizer
            #loss = tf.negative(maxi) + decay * regularizer
            losses.append(maxi)

        return tf.negative(tf.reduce_mean(tf.stack(losses)))










