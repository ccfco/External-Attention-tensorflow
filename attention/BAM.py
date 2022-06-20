import tensorflow as tf
from tensorflow.keras import layers, Sequential


class ChannelAttention(layers.Layer):
    def __init__(self, channel, reduction=16, num_layers=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = layers.GlobalAvgPool2D(keepdims=True)
        gate_channels = [channel]
        gate_channels += [channel//reduction] * num_layers
        gate_channels += [channel]

        self.ca = Sequential()
        for i in range(len(gate_channels)-2):
            self.ca.add(layers.Dense(gate_channels[i+1]))
            self.ca.add(layers.BatchNormalization())
            self.ca.add(layers.Activation('relu'))
        self.ca.add(layers.Dense(gate_channels[-1]))

    def call(self, x):
        res = self.avg_pool(x)
        res = self.ca(res)
        res = tf.broadcast_to(res, x.get_shape())
        return res

class SpatialAttention(layers.Layer):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super(SpatialAttention, self).__init__()
        self.sa = Sequential()
        self.sa.add(layers.Conv2D(filters=channel//reduction, kernel_size=1))
        self.sa.add(layers.BatchNormalization())
        self.sa.add(layers.Activation('relu'))
        for i in range(num_layers):
            self.sa.add(layers.Conv2D(filters=channel//reduction, kernel_size=3, padding='same', dilation_rate=dia_val))
            self.sa.add(layers.BatchNormalization())
            self.sa.add(layers.Activation('relu'))
        self.sa.add(layers.Conv2D(1, kernel_size=1))

    def call(self, x):
        res = self.sa(x)
        res = tf.broadcast_to(res, x.get_shape())
        return res

class BAMBlock(layers.Layer):
    def __init__(self, channel=512, reduction=16, dia_val=2):
        super(BAMBlock, self).__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = tf.sigmoid

    def call(self, x):
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out+ca_out)
        out = (1+weight)*x
        return out

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    bam = BAMBlock(channel=512, reduction=16, dia_val=2)
    output = bam(input)
    print(output.shape)
