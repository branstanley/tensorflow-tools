import tensorflow as tf

margin_var = tf.Variable(0.8, trainable=False)

def cosine_triplet_loss():
    def loss(y_true, y_pred):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
        pos_sim = tf.reduce_sum(anchor * positive, axis=1)
        neg_sim = tf.reduce_sum(anchor * negative, axis=1)
        return tf.reduce_mean(tf.maximum(0.0, neg_sim - pos_sim + margin_var))
    return loss
