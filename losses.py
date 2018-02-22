import tensorflow as tf

def bpr_loss(users, pos_items, neg_items, lambda_u, lambda_v):
    losses = []
    for u, pos, neg in zip(tf.unstack(users), tf.unstack(pos_items), tf.unstack(neg_items)):
        pos_score = tf.reduce_sum(tf.multiply(u, pos))
        neg_score = tf.reduce_sum(tf.multiply(u, neg))

        # Loss function using L2 Regularization
        l2_loss = lambda_u*tf.nn.l2_loss(u) + lambda_v*tf.nn.l2_loss(pos) + lambda_v*tf.nn.l2_loss(neg)
        maxi = tf.log(tf.nn.sigmoid(pos_score - neg_score)) - l2_loss
        # loss = tf.negative(maxi) + decay * regularizer
        losses.append(maxi)

    return tf.negative(tf.reduce_mean(tf.stack(losses)))