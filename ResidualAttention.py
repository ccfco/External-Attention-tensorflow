
import tensorflow as tf

class ResidualAttention():
    def __init__(self, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = tf.keras.layers.Conv2D(filters=num_class, kernel_size=1, strides=1, use_bias=False)

    def __call__(self, x):
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
