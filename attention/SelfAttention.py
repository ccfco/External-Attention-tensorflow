import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ScaledDotProductAttention(layers.Layer):
    """
    Scaled dot-product attention
    """
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = layers.Dense(h * d_k)
        self.fc_k = layers.Dense(h * d_k)
        self.fc_v = layers.Dense(h * d_k)
        self.fc_o = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def call(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computs
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = tf.reshape(self.fc_q(queries), (b_s, nq, self.h, self.d_k))
        q = tf.transpose(q, (0, 2, 1, 3))  # (b_s, h, nq ,d_k)
        k = tf.transpose(tf.reshape(self.fc_k(keys), (b_s, nk, self.h, self.d_k)), (0, 2, 3, 1))  # (b_s, h, d_k, nk)
        v = tf.transpose(tf.reshape(self.fc_v(values), (b_s, nk, self.h, self.d_v)), (0, 2, 1, 3))  # (b_s, h, nk ,d_v)

        att = tf.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = tf.nn.softmax(att, -1)
        att = self.dropout(att)

        out = tf.reshape(tf.transpose(tf.matmul(att, v), (0, 2, 1, 3)), (b_s, nq, self.h * self.d_v))  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (bs, nq, d_model)
        return out


if __name__ == '__main__':
    input = tf.random.normal((50, 49, 512))
    sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    output = sa(input, input, input)
    print(output.shape)
