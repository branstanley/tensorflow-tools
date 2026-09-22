import tensorflow as tf

def circle_loss(margin=0.25, gamma=64.0):
    def loss(y_true, y_pred):
        # Accept either tuple/list (a, p, n) or a single [B, 3*D] tensor
        if isinstance(y_pred, (tuple, list)) and len(y_pred) == 3:
            anchor, positive, negative = y_pred
        else:
            # y_pred is [B, 3*D] — split along the last axis
            last_dim = tf.shape(y_pred)[-1]
            # Ensure divisible by 3; this avoids silent shape bugs
            tf.debugging.assert_greater_equal(
                last_dim % 3, 0,
                message="Last dimension of y_pred must be divisible by 3 for concatenated [a|p|n]."
            )
            D = last_dim // 3
            anchor   = y_pred[:, :D]
            positive = y_pred[:, D:2*D]
            negative = y_pred[:, 2*D:]

        # Normalize for cosine similarity
        a = tf.math.l2_normalize(anchor, axis=-1)
        p = tf.math.l2_normalize(positive, axis=-1)
        n = tf.math.l2_normalize(negative, axis=-1)

        # Cosine similarities per sample [B]
        s_p = tf.reduce_sum(a * p, axis=-1)
        s_n = tf.reduce_sum(a * n, axis=-1)

        m = margin
        g = gamma

        # Circle weights and deltas
        alpha_p = tf.nn.relu(1.0 + m - s_p)   # positive weighting
        alpha_n = tf.nn.relu(s_n + m)         # negative weighting
        delta_p = 1.0 - m
        delta_n = m

        # Per-sample terms
        pos_term = tf.exp(g * alpha_p * (s_p - delta_p))    # [B]
        neg_term = tf.exp(g * alpha_n * (delta_n - s_n))    # [B]

        # Final per-sample loss and batch mean
        loss_i = tf.math.log1p(pos_term * neg_term)         # log(1 + pos * neg)
        return tf.reduce_mean(loss_i)
