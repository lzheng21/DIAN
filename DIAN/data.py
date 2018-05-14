import numpy as np
import random as rd


class Data(object):
    def __init__(self, train_file='data/ml-1m/train_users.dat', test_file='data/ml-1m/test_users.dat'):
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


        self.train_items, self.val_set, self.test_set = {}, {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, test_items = items[0], items[1:]
                    if len(test_items) <= 1:
                        self.test_set[uid] = test_items
                    else:
                        self.val_set[uid] = test_items[:int(len(test_items)/2)]
                        self.test_set[uid] = test_items[int(len(test_items)/2):]

    def sample_pairs(self, batch_size):
        if batch_size <= self.n_users:
            users = rd.sample(range(self.n_users), batch_size)
        else:
            users = [rd.choice(range(self.n_users)) for _ in range(batch_size)]


        pos_items = []
        neg_items = []
        for u in users:
            pos_items += self.sample_pos_item_for_u(u)
            neg_items += self.sample_neg_item_for_u(u)

        return users, pos_items, neg_items

    def sample(self, batch_size):
        users, items, labels = [], [], []
        if int(batch_size/2) <= self.n_users:
            uids = rd.sample(range(self.n_users), int(batch_size/2))
        else:
            uids = [rd.choice(range(self.n_users)) for _ in range(int(batch_size/2))]

        for u in uids:
            pos_item = self.sample_pos_item_for_u(u)
            neg_item = self.sample_neg_item_for_u(u)

            users, items, labels = users+[u], items+[pos_item], labels+[1]
            users, items, labels = users+[u], items+[neg_item], labels+[0]
        if len(users) < batch_size:
            uid = rd.choice(range(self.n_users))
            users, items, labels = users + [uid], items + [self.sample_pos_item_for_u(uid)], labels + [1]
        users = np.squeeze(np.asarray(users))
        items = np.squeeze(np.asarray(items))
        labels = np.squeeze(np.asarray(labels))
        return users, items, labels

    def sample_pos_item_for_u(self, u):
        pos_items = self.train_items[u]
        return rd.sample(pos_items, 1)

    def sample_neg_item_for_u(self, u):
        neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
        return rd.sample(neg_items, 1)

    def get_num_users_items(self):
        return self.n_users, self.n_items



