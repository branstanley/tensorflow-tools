import tensorflow as tf

margin_var = tf.Variable(0.15, trainable=False)

def combined_triplet_contrastive_loss(alpha=0.4, beta=0.1):
    def loss(y_true, y_pred):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
        pos_sim = tf.reduce_sum(anchor * positive, axis=1)
        neg_sim = tf.reduce_sum(anchor * negative, axis=1)

        # Triplet loss
        triplet_loss = tf.reduce_mean(tf.maximum(0.0, neg_sim - pos_sim + margin_var))

        # Contrastive loss (simplified)
        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim
        contrastive_loss = tf.reduce_mean(3.0*pos_dist**2 + tf.maximum(0.0, margin_var - neg_dist)**2)

        pos_pull = tf.reduce_mean(1.0 - pos_sim)
        total_loss = alpha * triplet_loss + (1 - alpha) * contrastive_loss + beta * pos_pull
        return total_loss
    return loss