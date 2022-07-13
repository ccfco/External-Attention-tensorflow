import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


class ShuffleAttention(layers.Layer):
    def __init__(self, channel=512, reduction=16, G=8):
        super(ShuffleAttention, self).__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = layers.GlobalAvgPool2D(keepdims=True)
        self.gn = tfa.layers.GroupNormalization(channel // (2 * G), axis=-1)
        self.sigmoid = tf.nn.sigmoid

    def build(self, input_shape):
        self.cweight = self.add_weight(
            shape=(1, 1, 1, self.channel // (2 * self.G)), initializer='zeros', trainable=True,
        )
        self.cbias = self.add_weight(
            shape=(1, 1, 1, self.channel // (2 * self.G)), initializer='ones', trainable=True,
        )
        self.sweight = self.add_weight(
            shape=(1, 1, 1, self.channel // (2 * self.G)), initializer='zeros', trainable=True,
        )
        self.sbias = self.add_weight(
            shape=(1, 1, 1, self.channel // (2 * self.G)), initializer='ones', trainable=True,
        )
        super(ShuffleAttention, self).build(input_shape)

    @staticmethod
    def channel_shuffle(x, groups):
        b, h, w, c = x.get_shape()
        x = tf.reshape(x, shape=(b, h, w, groups, -1))
        x = tf.transpose(x, perm=(0, 1, 2, 4, 3))

        # flatten
        x = tf.reshape(x, shape=(b, h, w, -1))
        return x

    def call(self, x):
        b, h, w, c = x.get_shape()
        # group into subfeatures
        x = tf.reshape(x, (b * self.G, h, w, -1))  # bs*G, h, w, c//G

        # channel_split
        x_0, x_1 = tf.split(x, num_or_size_splits=2, axis=3)  # bs*G, h, w, c//(2*G)

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G, 1, 1, c//(2*G)
        x_channel = self.cweight * x_channel + self.cbias  # bs*G, 1, 1, c//(2*G)
        x_channel = x_0 * self.sigmoid(x_channel)  # bs*G, h, w, c//(2*G)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G, h, w, c//(2*G)
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G, h, w, c//(2*G)
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G, h, w, c//(2*G)

        # concatenate along channel axis
        out = tf.concat([x_channel, x_spatial], axis=3)
        out = tf.reshape(out, (b, h, w, -1))

        # channel shuffle
        out = self.channel_shuffle(out, 2)

        return out


if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    se = ShuffleAttention(channel=512, G=8)
    output = se(input)
    print(output.shape)
