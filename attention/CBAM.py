import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    pass

class SpatialAttention(layers.Layer):
    pass

class CBAMBlock(layers.Layer):
    pass

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    kernel_size = input.get_shape()[1]
    cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
    output = cbam(input)
    print(output.shape)