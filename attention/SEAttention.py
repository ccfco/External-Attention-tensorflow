import tensorflow as tf
from tensorflow.keras import layers, Sequential

class SEAttention(layers.Layer):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = layers.GlobalAvgPool2D()  # 同nn.AdaptiveAvgPool2d(1)， 但是注意torch的输出是保持4维的,而tensorflow不保持维度.
        self.fc = Sequential([
            layers.Dense(channel // reduction, use_bias=False),
            layers.Activation('relu'),
            layers.Dense(channel, use_bias=False),
            layers.Activation('sigmoid')
        ])

    def call(self, x):
        b, h, w, c = x.get_shape()
        y = self.avg_pool(x)
        y = tf.expand_dims(tf.expand_dims(self.fc(y), axis=1), axis=2)
        return x * tf.tile(y, (1, h, w, 1))

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    se = SEAttention(channel=512, reduction=8)
    output = se(input)
    print(output.shape)

