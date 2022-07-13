import tensorflow as tf
from tensorflow.keras import layers, Sequential


class ChannelAttention(layers.Layer):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = layers.GlobalMaxPool2D(keepdims=True)
        self.avgpool = layers.GlobalAvgPool2D(keepdims=True)
        self.se = Sequential([
            layers.Conv2D(channel // reduction, 1, use_bias=False),
            layers.Activation('relu'),
            layers.Conv2D(channel, 1, use_bias=False)
        ])
        self.sigmoid = tf.sigmoid

    def call(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, padding='same')
        self.sigmoid = tf.sigmoid

    def call(self, x):
        max_result = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_result = tf.reduce_mean(x, axis=-1, keepdims=True)
        result = tf.concat([max_result, avg_result], -1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(layers.Layer):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        b, _, _, c = x.get_shape()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    kernel_size = input.get_shape()[1]
    cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
    output = cbam(input)
    print(output.shape)
