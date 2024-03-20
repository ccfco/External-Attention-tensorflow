import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Softmax, LayerNormalization, Activation
import tensorflow.keras.backend as K

class SequentialPolarizedSelfAttention(Layer):
    def __init__(self, channel=512, **kwargs):
        super(SequentialPolarizedSelfAttention, self).__init__(**kwargs)
        self.channel = channel
        self.ch_wv = Conv2D(channel // 2, kernel_size=(1, 1), padding='same')
        self.ch_wq = Conv2D(1, kernel_size=(1, 1), padding='same')
        self.softmax_channel = Softmax(axis=1)
        self.softmax_spatial = Softmax(axis=-1)
        self.ch_wz = Conv2D(channel, kernel_size=(1, 1), padding='same')
        self.ln = LayerNormalization(axis=[1, 2, 3])
        self.sigmoid = Activation('sigmoid')
        self.sp_wv = Conv2D(channel // 2, kernel_size=(1, 1), padding='same')
        self.sp_wq = Conv2D(channel // 2, kernel_size=(1, 1), padding='same')
        self.agp = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

    def call(self, x):
        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x) # bs, h, w, c//2
        channel_wq = self.ch_wq(x) # bs, h, w, 1
        channel_wv = tf.reshape(channel_wv, [tf.shape(x)[0], -1, self.channel // 2]) # bs, h*w, c//2
        channel_wq = tf.reshape(channel_wq, [tf.shape(x)[0], -1, 1]) # bs, h, w, 1
        channel_wq = self.softmax_channel(channel_wq) # bs, h*w, 1
        channel_wz = tf.matmul(channel_wv, channel_wq, transpose_a=True)  # bs, c//2, 1
        channel_wz = tf.reshape(channel_wz, [tf.shape(x)[0], 1, 1, self.channel // 2])
        channel_wz = self.ch_wz(channel_wz)
        channel_wz = tf.reshape(channel_wz, [tf.shape(x)[0], 1, 1, self.channel])
        channel_weight = self.sigmoid(self.ln(channel_wz)) # bs, 1, 1, c
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(channel_out) # bs, h, w, c//2
        spatial_wq = self.sp_wq(channel_out) # bs, h, w, c//2
        spatial_wq = self.agp(spatial_wq) # bs, 1, 1, c//2
        spatial_wv = tf.reshape(spatial_wv, [tf.shape(x)[0], -1, self.channel // 2]) # bs, h*w, c//2
        spatial_wq = tf.reshape(spatial_wq, [tf.shape(x)[0], 1, self.channel // 2]) # bs, 1, c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = tf.matmul(spatial_wq, spatial_wv, transpose_b=True) # bs, 1, h*w, 
        spatial_weight = self.sigmoid(tf.reshape(spatial_wz, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])) # bs, h, w, 1
        spatial_out = spatial_weight * channel_out
        
        return spatial_out

# Test the SequentialPolarizedSelfAttention layer
if __name__ == '__main__':
    input_tensor = tf.random.normal([1, 7, 7, 512])
    psa = SequentialPolarizedSelfAttention(channel=512)
    output_tensor = psa(input_tensor)
    print(output_tensor.shape)