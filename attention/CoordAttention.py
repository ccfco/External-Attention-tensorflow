import tensorflow as tf
from tensorflow.keras import layers

class CoordAtt(layers.Layer):
    pass

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    a2 = CoordAtt(512, 128, 128, True)
    output = a2(input)
    print(output.shape)