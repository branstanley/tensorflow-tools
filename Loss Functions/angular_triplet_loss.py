import tensorflow as tf
import math

margin_var = tf.Variable(0.8, trainable=False)

def angular_triplet_loss():
    def loss(y_true, y_pred):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
        pos_sim = tf.clip_by_value(tf.reduce_sum(anchor * positive, axis=1), -1.0, 1.0)
        neg_sim = tf.clip_by_value(tf.reduce_sum(anchor * negative, axis=1), -1.0, 1.0)
        pos_angle = tf.acos(pos_sim) / tf.constant(math.pi)
        neg_angle = tf.acos(neg_sim) / tf.constant(math.pi)
        return tf.reduce_mean(tf.maximum(0.0, pos_angle - neg_angle + margin_var))
    return loss