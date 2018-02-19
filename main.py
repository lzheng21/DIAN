import multiprocessing
import random

cores = multiprocessing.cpu_count()

from load_data import *
from models.BPR import *

#########################################################################################
# Hyper-parameters
#########################################################################################
MODEL = 'BPR'
EMB_DIM = 32
BATCH_SIZE = 1024
DECAY = 0.001
LAMDA = 1
K = 3
N_EPOCH = 50000
LR = 0.001

DIR = 'data/ml-1m'

NEGATIVE_SIZE = {'BPR': 1, 'GCN': 3}[MODEL]


data_generator = Data(DIR=DIR, batch_size=BATCH_SIZE, negative_size=NEGATIVE_SIZE)
USER_NUM, ITEM_NUM = data_generator.get_num_users_items()


def simple_test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    training_items = data_generator.train_items[u]
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])

    ndcg_3 = ut.ndcg_at_k(r, 3)

    recall_5 = ut.recall(item_sort, user_pos_test, 5)
    recall_10 = ut.recall(item_sort, user_pos_test, 10)
    recall_50 = ut.recall(item_sort, user_pos_test, 50)
    ap = ut.average_precision(r)


    return np.array([recall_5, recall_10, recall_50, ap])


def simple_test(sess, model, users_to_test):
    result = np.array([0.] * 4)
    pool = multiprocessing.Pool(cores)
    batch_size = BATCH_SIZE
    #all users needed to test
    test_users = users_to_test
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def main():
    if MODEL == 'BPR':
        model = BPR(n_users=USER_NUM, n_items=ITEM_NUM, emb_dim=EMB_DIM, lr=LR, batch_size=BATCH_SIZE, decay=DECAY)
    else:
        assert 'invalid model'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for epoch in range(N_EPOCH):
        users, items = data_generator.sample()
        _, loss = sess.run([model.updates, model.loss],
                                           feed_dict={model.users: users, model.items: items})

        users_to_test = random.sample(data_generator.val_set.keys(),BATCH_SIZE)

        val_ret = simple_test(sess, model, users_to_test)
        recall_5, recall_10, recall_50, mAP= \
            val_ret[0], val_ret[1], val_ret[2], val_ret[3]

        print('Epoch %d training loss %f val recall_5 %f val recall_10 %f val recall_50 %f  val mAP %f'
              % (epoch, loss, recall_5, recall_10, recall_50, mAP))

if __name__ == '__main__':
    main()