import math

import tensorflow as tf
from tensorflow.keras import layers


class OutlookAttention(layers.Layer):
    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False, attn_drop=0.1):
        super(OutlookAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = layers.Dense(dim, use_bias=qkv_bias)
        self.attn = layers.Dense(kernel_size ** 4 * num_heads)

        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(attn_drop)

        self.unflod = tf.image.extract_patches(sizes=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
                                               padding='same')
        self.pool = layers.AvgPool2D(pool_size=stride, strides=stride, ceil_mode=True)

    def call(self, x):
        B, H, W, C = x.get_shape()

        # 映射到新的特征v
        v = self.v_pj(x)
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = tf.reshape(self.unflod(v), (B, self.num_heads, h * w, self.kernel_size * self.kernel_size, self.head_dim))

        # 生成Attention Map
        attn = self.pool(x)
        attn = tf.reshape(self.attn(attn),
                          (B, self.num_heads, h * w, self.kernel_size * self.kernel_size,
                           self.kernel_size * self.kernel_size))

        attn = self.scale * attn
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # 获取weighted特征
        out = tf.reshape((attn @ v), (B, h*w, C * self.kernel_size * self.kernel_size))
        out = tf.fold  # torch.nn.fold开发者说没有这个功能，未来没打算加，以后再补充。见https://github.com/tensorflow/tensorflow/issues/52195#issuecomment-948915934


if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    outlook = OutlookAttention(512, 128, 128, True)
    output = outlook(input)
    print(output.shape)
