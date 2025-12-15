from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Add, Activation
)
import tensorflow as tf

def residual_block(x, units, dropout_rate=0.4, l2_reg=0.0003):
    shortcut = x

    # First dense layer
    x = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    # Second dense layer
    x = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)

    # Projection shortcut if dimensions differ
    if shortcut.shape[-1] != units:
        shortcut = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(shortcut)

    # Add skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x