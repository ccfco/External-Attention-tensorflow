import tensorflow as tf
from tensorflow.keras import layers

class HaloAttention(layers.Layer):
    pass

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    halo = HaloAttention(512, 128, 128, True)
    output = halo(input)
    print(output.shape)
    # 参考https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/halonet/halonet.py