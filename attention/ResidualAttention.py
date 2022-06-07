import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model

class ResidualAttention(Model):
    def __init__(self, num_class=1000, name='ResidualAttention', la=0.2):
        super(ResidualAttention, self).__init__(name=name)
        self.la = la
        self.fc = Conv2D(filters=num_class, kernel_size=1, strides=1, use_bias=False)

    def call(self, x):
        x = self.fc(x)
        b, h, w, c = x.shape
        y_raw = tf.reshape(x, [-1, h*w, c]) #b, hxw, num_class
        y_avg = tf.reduce_mean(y_raw, axis=1) #b, num_class
        y_max = tf.reduce_max(y_raw, axis=1) #b, num_class
        score = y_avg+self.la*y_max
        return score

if __name__ == '__main__':
    input = tf.random.normal(shape=(50, 7, 7, 512))
    resatt = ResidualAttention(num_class=1000, la=0.2)
    output = resatt(input)
    print(output.shape)
