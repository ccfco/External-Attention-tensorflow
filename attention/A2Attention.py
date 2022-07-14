import tensorflow as tf
from tensorflow.keras import layers


class DoubleAttention(layers.Layer):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super(DoubleAttention, self).__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = layers.Conv2D(c_m, 1)
        self.convB = layers.Conv2D(c_n, 1)
        self.convV = layers.Conv2D(c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = layers.Conv2D(in_channels, kernel_size=1)

    def call(self, x):
        b, h, w, c = x.get_shape()
        assert c == self.in_channels
        A = self.convA(x)  # b, h, w, c_m
        B = self.convB(x)  # b, h, w, c_n
        V = self.convV(x)  # b, h, w, c_n
        tmpA = tf.reshape(A, (b, self.c_m, -1))
        attention_maps = tf.nn.softmax(tf.reshape(B, (b, -1, self.c_n)))
        attention_vectors = tf.nn.softmax(tf.reshape(V, (b, self.c_n, -1)))
        # step 1: feature gating
        global_descriptors = tf.matmul(tmpA, attention_maps)  # b, c_m, c_n
        # step 2: feature distribution
        tmpZ = tf.matmul(global_descriptors, attention_vectors)  # b, c_m, h*w
        tmpZ = tf.reshape(tmpZ, (b, h, w, self.c_m))  # b, h, w, c_m
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    a2 = DoubleAttention(512, 128, 128, True)
    output = a2(input)
    print(output.shape)
