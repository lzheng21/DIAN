import numpy as np
import random as rd


class Data(object):
    def __init__(self, batch_size, train_file='data/ml-1m/train_users.dat', test_file='data/ml-1m/test_users.dat'):
        self.batch_size = batch_size


        #get number of users and items
        self.n_users, self.n_items = 0, 0


        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    self.n_users += 1
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))
        self.n_items += 1
        self.graph = np.zeros((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.graph[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def sample_pairs(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(range(self.n_users), self.batch_size)
        else:
            users = [rd.choice(range(self.n_users)) for _ in range(self.batch_size)]

        def sample_pos_item_for_u(u):
            pos_items = self.train_items[u]
            return rd.sample(pos_items, 1)

        def sample_neg_item_for_u(u):
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            return rd.sample(neg_items, 1)

        pos_items = []
        neg_items = []
        for u in users:
            pos_items += sample_pos_item_for_u(u)
            neg_items += sample_neg_item_for_u(u)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items



