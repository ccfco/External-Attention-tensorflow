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
        self.softmax = tf.nn.softmax

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int):
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.h, self.d_model))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

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

        q = self.transpose_for_scores(self.fc_q(queries), batch_size=b_s)  # (b_s, h, nq ,d_k)
        k = self.transpose_for_scores(self.fc_k(keys), batch_size=b_s)  # (b_s, h, nk, d_k)
        v = self.transpose_for_scores(self.fc_v(values), batch_size=b_s)  # (b_s, h, nk ,d_v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        att = tf.matmul(q, k, transpose_b=True) / np.sqrt(self.d_k)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = tf.multiply(att, attention_mask)

        # Normalize the attention scores to probabilities.
        att = self.softmax(att, -1)

        att = self.dropout(att)

        out = tf.reshape(tf.transpose(tf.matmul(att, v), (0, 2, 1, 3)),
                         (b_s, nq, self.h * self.d_v))  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


if __name__ == '__main__':
    input = tf.random.normal((50, 49, 512))
    sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    output = sa(input, input, input)
    print(output.shape)
