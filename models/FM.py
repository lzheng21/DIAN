import tensorflow as tf
import utils as ut
import numpy as np
from losses import *
class FM(object):
    def __init__(self, data, emb_dim, batch_size, lambda_u, lambda_v):
        self.data = data
        self.n_users, self.n_items = self.data.get_num_users_items()
        self.emb_dim = emb_dim
        self.batch_size = batch_size


        #TO DO

    def train(self, n_epoch, lr):
        self.opt = tf.train.AdamOptimizer(lr)
        self.updates = self.opt.minimize(self.loss, var_list=self.params)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):
            users, pos_items, neg_items = self.data.sample_pairs()
            _, loss = self.sess.run([self.updates, self.loss],
                                   feed_dict={self.users: users,
                                              self.pos_items: pos_items,
                                              self.neg_items: neg_items})

            print('\rEpoch %d training loss %f' % (epoch, loss), end='')

    def predict(self):
        result = np.array([0.] * 6)
        import multiprocessing
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cores)

        # all users needed to test
        test_users = self.data.test_set.keys()
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
        p_3, ndcg, recall_5, recall_10, recall_50, mAP = \
            ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]

        print(
            'val p_3 %f val ndcg %f val recall_5 %f val recall_10 %f val recall_50 %f  val mAP %f'
            % (p_3, ndcg, recall_5, recall_10, recall_50, mAP))












