
import tensorflow as tf

margin_var = tf.Variable(0.15, trainable=False, dtype=tf.float32, name="margin")
gamma_var = tf.Variable(64.0, trainable=False, dtype=tf.float32, name="gamma")
lambda_weight_var = tf.Variable(0.5, trainable=False, dtype=tf.float32, name="lambda_weight")

def hybrid_circle_cosine_triplet_loss():
    def loss(y_true, y_pred):
        last_dim = tf.shape(y_pred)[-1]
        # Optional safety: assert divisibility by 3
        tf.debugging.assert_equal(
            (last_dim % 3), 0,
            message="y_pred's last dimension must be divisible by 3 for [anchor|positive|negative]."
        )
        D = last_dim // 3
        anchor   = y_pred[:, :D]
        positive = y_pred[:, D:2*D]
        negative = y_pred[:, 2*D:]

        # --- normalize for cosine similarities ---
        a = tf.math.l2_normalize(anchor, axis=-1)
        p = tf.math.l2_normalize(positive, axis=-1)
        n = tf.math.l2_normalize(negative, axis=-1)

        # --- cosine similarities per sample ---
        s_p = tf.reduce_sum(a * p, axis=-1)  
        s_n = tf.reduce_sum(a * n, axis=-1) 

        # Cast hyperparams to the same dtype as sims (float32)
        margin = tf.cast(margin_var, s_p.dtype)
        gamma  = tf.cast(gamma_var,  s_p.dtype)
        lam    = tf.cast(lambda_weight_var, s_p.dtype)

        # --- Triplet (cosine hinge) per-sample then mean ---
        # L_triplet_i = relu(margin + s_n - s_p)
        triplet_per = tf.nn.relu(margin + s_n - s_p)  
        triplet_loss = tf.reduce_mean(triplet_per)

        # --- Circle Loss (cosine), per-sample aggregation ---
        # alpha_p = relu(1 + m - s_p), alpha_n = relu(s_n + m)
        alpha_p = tf.nn.relu(1.0 + margin - s_p)
        alpha_n = tf.nn.relu(s_n + margin)
        delta_p = 1.0 - margin
        delta_n = margin

        # One positive & one negative per anchor -> no sum across batch
        pos_term = tf.exp(gamma * alpha_p * (s_p - delta_p))   
        neg_term = tf.exp(gamma * alpha_n * (delta_n - s_n))  
        circle_per = tf.math.log1p(pos_term * neg_term)        
        circle_loss = tf.reduce_mean(circle_per)

        # --- Hybrid ---
        total_loss = lam * triplet_loss + (1.0 - lam) * circle_loss
        return total_loss
    return loss
