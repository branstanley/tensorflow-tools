from ModelBuildingBlocks.residual_block import residual_block
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

def build_encoder(num_classes, features, input_shape=(650,)):
    norm = layers.Normalization(axis=-1)

    dropout_rate = 0.5
    l2_reg = 1e-3

    input = layers.Input(shape=input_shape, name="source_1")
    x = norm(input)
    # Initial projection
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    # Residual blocks with projection shortcuts
    x = residual_block(x, 512, dropout_rate=dropout_rate, l2_reg=l2_reg)
    x = residual_block(x, 256, dropout_rate=dropout_rate, l2_reg=l2_reg)
    # Final dense layer before output
    embedding = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)

    output = tf.keras.layers.Dense(num_classes, activation='softmax')(embedding)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    norm.adapt(features)

    return model