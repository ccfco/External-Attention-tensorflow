import tensorflow as tf
from tensorflow.keras import layers

def drop_connect(inputs, p, training):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not training: return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = inputs / keep_prob * binary_tensor
    return output

class MBConvBlock(layers.Layer):
    """A class of MBVonv: Mobile Inverted Residual Bottleneck.
    Attributes:
        endpoints: dict. A list of internal tensors.
        层：ksize=3*3 输入32 输出16 conv1 stride1
    """

    def __init__(self, ksize, input_filters, output_filters, expand_ratio=1, stride=1, name=None):
        super().__init__(name=name)

        self._bn_mom = 0.1  # batch norm momentum
        self._bn_eps = 0.1  # batch norm epsilon
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self.kernel_size = ksize
        self._stride = stride

        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            self._expand_conv = layers.Conv2D(filters=oup, kernel_size=1, padding='same', use_bias=False)
            self._bn0 = layers.BatchNormalization(momentum=self._bn_mom, epsilon=self._bn_eps)

        # Depthwise convolution
        k = self.kernel_size
        s = self._stride
        self._depthwise_conv = layers.Conv2D(filters=oup, groups=oup, kernel_size=k, strides=s, padding='same',
                                             use_bias=False)
        self._bn1 = layers.BatchNormalization(momentum=self._bn_mom, epsilon=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        num_squeezed_channels = max(1, self._input_filters * self._se_ratio)  # num reduced filters
        self._se_reduce = layers.Conv2D(filters=num_squeezed_channels, kernel_size=1, padding='same')
        self._se_expand = layers.Conv2D(filters=oup, kernel_size=1, padding='same')

        # Output phase
        final_oup = self._output_filters
        self._project_conv = layers.Conv2D(filters=final_oup, kernel_size=1, padding='same', use_bias=False)
        self._bn2 = layers.BatchNormalization(momentum=self._bn_mom, epsilon=self._bn_eps)
        self._swish = tf.nn.swish  # Swish 是一种新型激活函数，公式为： f(x) = x · sigmoid(x)

    def call(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(x)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)

        # Squeeze and Excitation
        h_axis, w_axis = [1, 2]
        x_squeezed = tf.nn.avg_pool2d(x, ksize=[1, x.shape[h_axis], x.shape[w_axis], 1], strides=[1, 1, 1, 1],
                                      padding='VALID')
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = tf.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if drop_connect_rate is not None:
                x = drop_connect(x, p=drop_connect_rate, training=True)
            x = x + inputs  # skip connection
        return x

if __name__ == '__main__':
    input = tf.random.normal((1, 112, 112, 3))
    mbconv = MBConvBlock(ksize=3, input_filters=3, output_filters=3)
    out = mbconv(input)
    print(out.shape)
