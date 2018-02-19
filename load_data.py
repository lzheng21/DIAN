import numpy as np
import random as rd


class Data(object):
    def __init__(self, DIR, batch_size, negative_size):
        self.DIR = DIR
        self.batch_size = batch_size
        self.negative_size = negative_size


        #get number of users and items
        self.n_users, self.n_items = 0, 0
        with open(self.DIR + '/users.dat') as f:
            for l in f.readlines():
                if len(l) > 0:
                    self.n_users += 1
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))
            f.close()
        self.n_items += 1
        self.graph = np.zeros((self.n_users, self.n_items), dtype=np.float32)


        self.train_items, self.val_set, self.test_set = {}, {}, {}

        def split(item_list):
            train = item_list[:int(len(item_list)*0.8)]
            test = item_list[int(len(item_list)*0.8):]
            if len(test) > 1:
                val, test = test[:int(len(test)/2)], test[int(len(test)/2):]
            else:
                val = []
            return train, val, test

        with open(self.DIR + '/users.dat') as f:
            u = 0
            for l in f.readlines():
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')[1:]]
                if len(items) <= 5:
                    self.train_items[u] = items
                    for i in items:
                        self.graph[u][i] = 1
                    u += 1
                    continue
                train_items, val_items, test_items = split(items)
                for i in train_items:
                    self.graph[u][i] = 1

                self.train_items[u] = train_items

                self.val_set[u], self.test_set[u] = val_items, test_items
                u += 1
            f.close()


    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(range(self.n_users), self.batch_size)
        else:
            users = [rd.choice(range(self.n_users)) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = np.nonzero(self.graph[u,:])[0].tolist()
            if len(pos_items) >= num:
                return rd.sample(pos_items, num)
            else:
                return [rd.choice(pos_items) for _ in range(num)]

        def sample_neg_items_for_u(u):
            neg_items = np.nonzero(self.graph[u,:] == 0)[0].tolist()
            return rd.sample(neg_items, self.negative_size)

        items = []
        for u in users:
            items += sample_pos_items_for_u(u, 1) + sample_neg_items_for_u(u)

        return users, items

    def get_num_users_items(self):
        return self.n_users, self.n_items



