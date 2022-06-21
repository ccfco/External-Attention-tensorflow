import tensorflow as tf
from tensorflow.keras import layers

class ECAAttention(layers.Layer):
    def __init__(self, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.gap = layers.GlobalAvgPool2D()
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, padding='same')
        self.sigmoid = tf.sigmoid

    def call(self, x):
        y = self.gap(x)  # bs, 1, 1, c
        y = tf.expand_dims(y, -1)  # bs, c, 1
        y = self.conv(y)  # bs, c, 1
        y = self.sigmoid(y)  # bs, c, 1
        y = tf.transpose(tf.expand_dims(y, -1), (0, 2, 3, 1))  # bs, 1, 1, c
        return x * tf.broadcast_to(y, x.get_shape())

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    eca = ECAAttention(kernel_size=3)
    output = eca(input)
    print(output.shape)
