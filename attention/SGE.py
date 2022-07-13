import tensorflow as tf
from tensorflow.keras import layers


class SpatialGroupEnhance(layers.Layer):
    def __init__(self, groups):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = layers.GlobalAvgPool2D(keepdims=True)
        self.sig = tf.sigmoid

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(1, 1, 1, self.groups), initializer='zeros', trainable=True)
        self.bias = self.add_weight(shape=(1, 1, 1, self.groups), initializer='zeros', trainable=True)
        super(SpatialGroupEnhance, self).build(input_shape)

    def call(self, x):
        b, h, w, c = x.get_shape()
        x = tf.reshape(x, (b * self.groups, h, w, -1))  # bs*g, h, w, dim//g
        xn = x * self.avg_pool(x)  # bs*g, h, w, dim//g
        xn = tf.reduce_sum(xn, axis=-1, keepdims=True)  # bs*g, h, w, 1
        t = tf.reshape(xn, (b * self.groups, -1))  # bs*g, h*w

        t = t - tf.reduce_mean(t, axis=-1, keepdims=True)  # bs*g, h*w
        std = tf.math.reduce_std(t, axis=-1, keepdims=True) + 1e-5
        t = t / std  # bs*g, h*w
        t = tf.reshape(t, (b, h, w, self.groups))  # bs, h, w, g

        t = t * self.weight + self.bias  # bs, h, w, g
        t = tf.reshape(t, (b * self.groups, h, w, 1))  # bs*g, h, w, 1
        x = x * self.sig(t)  # bs*g, h, w, dim//g
        x = tf.reshape(x, (b, h, w, c))

        return x


if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    sge = SpatialGroupEnhance(groups=8)
    output = sge(input)
    print(output.shape)
