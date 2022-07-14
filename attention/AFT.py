import tensorflow as tf
from tensorflow.keras import layers


class AFT_FULL(layers.Layer):
    def __init__(self, d_model, n=49, simple=False):

        super(AFT_FULL, self).__init__()
        self.fc_q = layers.Dense(d_model)
        self.fc_k = layers.Dense(d_model)
        self.fc_v = layers.Dense(d_model)
        if simple:
            self.position_biases = tf.zeros((n, n))
        else:
            self.position_biases = tf.Variable(tf.ones((n, n)), trainable=True)
        self.d_model = d_model
        self.n = n
        self.sigmoid = tf.sigmoid

    def call(self, input):
        bs, n, dim = input.get_shape()

        q = self.fc_q(input)  # bs, n, dim
        k = tf.expand_dims(self.fc_k(input), axis=0)  # 1, bs, n, dim
        v = tf.expand_dims(self.fc_v(input), axis=0)  # 1, bs, n, dim
        numerator = tf.reduce_sum(tf.exp(k + tf.reshape(self.position_biases, (n, 1, -1, 1))) * v, 2)  # n, bs, dim
        denominator = tf.reduce_sum(tf.exp(k + tf.reshape(self.position_biases, (n, 1, -1, 1))), 2)  # n, bs, dim

        out = (numerator / denominator)  # n, bs, dim
        out = self.sigmoid(q) * (tf.transpose(out, (1, 0, 2)))

        return out


if __name__ == '__main__':
    input = tf.random.normal((50, 49, 512))
    aft_full = AFT_FULL(d_model=512, n=49)
    output = aft_full(input)
    print(output.shape)
