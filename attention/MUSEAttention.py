import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Depth_Pointwise_Conv1d(layers.Layer):
    def __init__(self, in_ch, out_ch, k):
        super(Depth_Pointwise_Conv1d, self).__init__()
        if k == 1:
            self.depth_conv = tf.identity
        else:
            self.depth_conv = layers.Conv1D(
                filters=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding='same'
            )
        self.pointwise_conv = layers.Conv1D(
            filters=out_ch,
            kernel_size=1,
            groups=1
        )
    def call(self, x):
        depth_conv_out = self.depth_conv(x)
        out = self.pointwise_conv(depth_conv_out)
        return out

class MUSEAttention(layers.Layer):
    def __init__(self, d_model, d_k, d_v, h, dropout=1):
        super(MUSEAttention, self).__init__()
        self.fc_q = layers.Dense(h * d_k)
        self.fc_k = layers.Dense(h * d_k)
        self.fc_v = layers.Dense(h * d_v)
        self.fc_o = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = tf.Variable(tf.ones(3))
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

        # Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

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

        v2 = tf.reshape(tf.transpose(v, (0, 2, 1, 3)), (b_s, nk, -1))  # bs, dim, nk
        self.dy_paras = tf.Variable(self.softmax(self.dy_paras, -1))

        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        # out2 = tf.transpose(out2, (0, 2, 1))  # bs, n, dim

        out = out + out2
        return out

if __name__ == '__main__':
    input = tf.random.normal((50, 49, 512))
    sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
    output = sa(input, input, input)
    print(output.shape)

