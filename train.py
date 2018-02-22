import tensorflow as tf
def train(model, data_generator, n_epoch):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if model.name == 'BPR':
        for epoch in range(n_epoch):
            users, pos_items, neg_items = data_generator.sample_pairs()
            _, loss = sess.run([model.updates, model.loss],
                                           feed_dict={model.users: users,
                                                      model.pos_items: pos_items,
                                                      model.neg_items: neg_items})



            print('\rEpoch %d training loss %f' % (epoch, loss), end='')
    return sess