import tensorflow as tf
from ModelBuildingBlocks.TensorFlowL2Normalization import L2Normalization

def add_projection_head(encoder, input_shape=(650,)):
    proj_input = tf.keras.layers.Input(shape=(128,))
    p = tf.keras.layers.Dense(128, activation='relu')(proj_input)
    p = tf.keras.layers.Dropout(0.4)(p)
    p = tf.keras.layers.Dense(128)(p)
    p = L2Normalization()(p)
    projection_head = tf.keras.Model(inputs=proj_input, outputs=p, name="projection_head1")

    # Combine encoder + projection head
    input_final = tf.keras.layers.Input(shape=input_shape)
    frozen_embedding_1 = encoder(input_final)
    embedding = projection_head(frozen_embedding_1)
    final_model = tf.keras.Model(inputs=input_final, outputs=embedding, name="final_model_1")

    return final_model