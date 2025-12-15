from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable()
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
