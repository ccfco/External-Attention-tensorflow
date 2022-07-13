import tensorflow as tf
from tensorflow.keras import layers, Sequential


class SKAttention(layers.Layer):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super(SKAttention, self).__init__()
        self.d = max(L, channel // reduction)
        self.convs = []
        # self.convs = Sequential([])
        for k in kernels:
            self.convs.append(
                Sequential([
                    layers.Conv2D(channel, kernel_size=k, padding='same', groups=group, name='conv'),
                    layers.BatchNormalization(name='bn'),
                    layers.Activation('relu', name='relu'),
                ])
            )
        self.fc = layers.Dense(self.d)
        self.fcs = []
        for i in range(len(kernels)):
            self.fcs.append(layers.Dense(channel))

    def call(self, x):
        bs, _, _, c = x.get_shape()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = tf.stack(conv_outs, 0)  # k, bs, h, w, channel

        ### fuse
        U = sum(conv_outs)  # bs, h, w, c

        ### reduction channel
        S = tf.reduce_mean(tf.reduce_mean(U, axis=-2), axis=-2)  # bs, c
        Z = self.fc(S)  # bs, d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(tf.reshape(weight, (bs, 1, 1, c)))  # bs, channel
        attention_weughts = tf.stack(weights, 0)  # k, bs, 1, 1, channel
        attention_weughts = tf.nn.softmax(attention_weughts, axis=0)  # k, bs, 1, 1, channel

        ### fuse
        V = tf.reduce_sum(attention_weughts * feats, 0)
        return V


if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    se = SKAttention(channel=512, reduction=8)
    output = se(input)
    print(output.shape)
