import tensorflow as tf
from tensorflow.keras import layers

class SKAttention(layers.Layer):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super(SKAttention, self).__init__()
        self.d =
