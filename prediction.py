import utils as ut
import numpy as np
import multiprocessing

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    data_generator = x[2]
    #user u's items in the training set
    training_items = data_generator.train_items[u]
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(data_generator.n_items))

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


    return np.array([p_3, ndcg_3, recall_5, recall_10, recall_50, ap])


def predict(sess, model, data_generator):
    result = np.array([0.] * 6)
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)
    batch_size = data_generator.batch_size
    #all users needed to test
    test_users = data_generator.test_set.keys()
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size
        FLAG = False
        if len(user_batch) < batch_size:
            user_batch += [0] * (batch_size - len(user_batch))
            user_batch_len = len(user_batch)
            FLAG = True
        user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch, [data_generator]*batch_size)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
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
