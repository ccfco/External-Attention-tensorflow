from math import sqrt

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from attention.SelfAttention import ScaledDotProductAttention
from conv.MBConv import MBConvBlock


class CoAtNet(layers.Layer):
    def __init__(self, in_ch, out_chs=[64, 96, 192, 384, 768]):
        super(CoAtNet, self).__init__()
        self.out_chs = out_chs
        self.maxpool2d = layers.MaxPool2D(pool_size=2, strides=2)
        self.maxpool1d = layers.MaxPool1D(pool_size=2, strides=2)

        self.s0 = Sequential([
            layers.Conv2D(in_ch, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(in_ch, kernel_size=3, padding='same')
        ])

        self.mlp0 = Sequential([
            layers.Conv2D(out_chs[0], kernel_size=1, padding='same', activation='relu'),
            layers.Conv2D(out_chs[0], kernel_size=1, padding='same')
        ])

        self.s1 = MBConvBlock(ksize=3, input_filters=out_chs[0], output_filters=out_chs[0])
        self.mlp1 = Sequential([
            layers.Conv2D(out_chs[1], kernel_size=1, activation='relu'),
            layers.Conv2D(out_chs[1], kernel_size=1, )
        ])

        self.s2 = MBConvBlock(ksize=3, input_filters=out_chs[1], output_filters=out_chs[1])
        self.mlp2 = Sequential([
            layers.Conv2D(out_chs[2], kernel_size=1, activation='relu'),
            layers.Conv2D(out_chs[2], kernel_size=1, )
        ])

        self.s3 = ScaledDotProductAttention(out_chs[2], out_chs[2] // 8, out_chs[2] // 8, 8)
        self.mlp3 = Sequential([
            layers.Dense(out_chs[3], activation='relu'),
            layers.Dense(out_chs[3])
        ])

        self.s4 = ScaledDotProductAttention(out_chs[3], out_chs[3] // 8, out_chs[3] // 8, 8)
        self.mlp4 = Sequential([
            layers.Dense(out_chs[4], activation='relu'),
            layers.Dense(out_chs[4])
        ])

    def call(self, x):
        B, H, W, C = x.get_shape()
        # stage0
        y = self.mlp0(self.s0(x))
        y = self.maxpool2d(y)
        # stage1
        y = self.mlp1(self.s1(y))
        y = self.maxpool2d(y)
        # stage2
        y = self.mlp2(self.s2(y))
        y = self.maxpool2d(y)
        # stage3
        y = tf.reshape(y, (B, -1, self.out_chs[2]))  # B, N, C
        y = self.mlp3(self.s3(y, y, y))
        y = self.maxpool1d(y)
        # stage4
        y = self.mlp4(self.s4(y, y, y))
        y = self.maxpool1d(y)
        N = y.get_shape()[-2]
        y = tf.reshape(y, (B, int(sqrt(N)), int(sqrt(N)), self.out_chs[4]))

        return y


if __name__ == '__main__':
    input = tf.random.normal((1, 224, 224, 3))
    coatnet = CoAtNet(3)
    output = coatnet(input)
    print(output.shape)
