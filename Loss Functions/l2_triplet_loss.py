import tensorflow as tf

def l2_triplet_loss(margin=0.2):
    def loss(y_true, y_pred):
        anchor, positive, negative = tf.split(y_pred, 3, axis=1)
        # Squared Euclidean distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(0.0, pos_dist - neg_dist + margin))
    return
