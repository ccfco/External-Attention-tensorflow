import tensorflow as tf
from tensorflow.keras import layers
from attention.SelfAttention import ScaledDotProductAttention
from  attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention

class PositionAttentionModule(layers.Layer):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super(PositionAttentionModule, self).__init__()
        self.cnn = layers.Conv2D(d_model, kernel_size=kernel_size, padding='same')
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def call(self, x):
        bs, h, w, c = x.get_shape()
        y = self.cnn(x)
        y = tf.reshape(y, shape=(bs, h*w, c))
        y = self.pa(y, y, y)  # bs, h*w, c
        return y

class ChannelAttentionModule(layers.Layer):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super(ChannelAttentionModule, self).__init__()
        self.cnn = layers.Conv2D(d_model, kernel_size=kernel_size, padding='same')
        self.pa = SimplifiedScaledDotProductAttention(H*W, h=1)

    def call(self, x):
        bs, h, w, c = x.get_shape()
        y = self.cnn(x)
        y = tf.reshape(y, shape=(bs, c, -1))  # bs, c, h*w
        y = self.pa(y, y, y)  # bs, c, h*w
        return y

class DAModule(layers.Layer):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super(DAModule, self).__init__()
        self.position_attention_module = PositionAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
        self.channel_attention_module = ChannelAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)

    def call(self, input):
        bs, h, w, c = input.get_shape()
        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)
        p_out = tf.reshape(p_out, shape=(bs, h, w, c))
        c_out = tf.reshape(tf.transpose(c_out, perm=[0, 2, 1]), shape=(bs, h, w, c))
        return p_out+c_out

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    danet = DAModule(d_model=512, kernel_size=3, H=7, W=7)
    print(danet(input).shape)