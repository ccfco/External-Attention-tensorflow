import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential


class EMSA(layers.Layer):
    def __init__(self, d_model, d_k, d_v, h, droupout=.1, H=7, W=7, ratio=3, apply_transform=True):
        super(EMSA, self).__init__()
        self.H = H
        self.W = W
        self.fc_q = layers.Dense(h * d_k)
        self.fc_k = layers.Dense(h * d_k)
        self.fc_v = layers.Dense(h * d_v)
        self.fc_o = layers.Dense(d_model)
        self.dropout = layers.Dropout(droupout)

        self.ratio = ratio
        if self.ratio > 1:
            self.sr = Sequential()
            self.sr_conv = layers.Conv2D(d_model, kernel_size=ratio + 1, strides=ratio, padding='same', groups=d_model)
            self.sr_ln = layers.LayerNormalization()

        self.apply_transform = apply_transform and h > 1
        if self.apply_transform:
            self.transform = Sequential()
            self.transform.add(layers.Conv2D(h, kernel_size=1, strides=1, data_format='channels_first'))
            self.transform.add(layers.Activation(tf.nn.softmax))
            '''
            Batch Normalisation(axis是沿着channel): 就是强行将数据拉回到均值为0，方差为1的正太分布上，这样不仅数据分布一致，而且避免发生梯度消失。依赖于batch的大小和输入sequence的深度。
            Layer Normalisation(axis是沿着batch): LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作。LN用于RNN效果比较明显，但是在CNN上，不如BN。
            Instance Normalisation(axis是沿着batch和channel): 同BN注重对每个batch进行归一化，保证数据分布一致，因为判别模型中结果取决于数据整体分布。但是图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
            Group Normalization: 主要是针对Batch Normalization对小batchsize效果差，GN将channel方向分group，然后每个group内做归一化，算(C//G)*H*W的均值，这样与batchsize无关，不受其约束。
            '''
            self.transform.add(layers.BatchNormalization(axis=[0, 1]))  # InstanceNormalisation，[0, 1] is bs and c.
            # self.transform.add(tfa.layers.InstanceNormalization())

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

        b_s, nq, c = queries.get_shape()

        q = self.transpose_for_scores(self.fc_q(queries), batch_size=b_s)  # (b_s, h, nq, d_k)

        if self.ratio > 1:
            x = tf.reshape(queries, shape=[b_s, self.H, self.W, c])  # (b_s, H, W, c)
            x = self.sr_conv(x)  # (b_s, h, w, c)
            x = tf.reshape(x, shape=[b_s, -1, c])  # (bs, n', c)
            x = self.sr_ln(x)
            k = self.transpose_for_scores(self.fc_k(x), batch_size=b_s)  # (bs, h, n', d_k)
            v = self.transpose_for_scores(self.fc_v(x), batch_size=b_s)  # (bs, h, n', d_v)
        else:
            k = self.transpose_for_scores(self.fc_k(keys), batch_size=b_s)  # (bs, h, nk, d_k)
            v = self.transpose_for_scores(self.fc_v(values), batch_size=b_s)  # (bs, h, nk, d_v)

        if self.apply_transform:
            att = tf.matmul(q, k, transpose_b=True) / np.sqrt(self.d_k)  # (bs, h, nq, n')
            att = self.transform(att)
        else:
            att = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(self.d_k)  # (bs, h, nq, n')
            att = tf.math.softmax(att, -1)  # (bs, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = tf.multiply(att, attention_mask)

        att = self.dropout(att)

        out = tf.reshape(tf.transpose(tf.matmul(att, v), perm=[0, 2, 1, 3]),
                         shape=(b_s, nq, self.h * self.d_v))  # (bs, nq, h*d_v)
        out = self.fc_o(out)
        return out


if __name__ == '__main__':
    input = tf.random.normal((50, 64, 512))
    emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
    output = emsa(input, input, input)
    print(output.shape)
