import numpy as np
import torch
from torch import nn
from torch.nn import init
import tensorflow as tf

from tensorflow.keras import layers

class ExternalAttention(layers.Layer):

    def __init__(self, d_model, S=64):
        super(ExternalAttention, self).__init__(name='ExternalAttention')
        self.mk = layers.Dense(S)
        self.mv = layers.Dense(d_model)

    def call(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = tf.nn.softmax(attn, axis=1)  # bs,n,S
        attn = attn / tf.reduce_sum(attn, axis=2, keepdims=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out

if __name__ == '__main__':
    input = tf.random.normal(shape=(50, 49, 512))
    ea = ExternalAttention(d_model=512, S=8)
    output = ea(input)
    print(output.shape)


