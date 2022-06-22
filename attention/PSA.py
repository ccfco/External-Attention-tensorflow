import tensorflow as tf
from tensorflow.keras import layers, Sequential

class PSA(layers.Layer):
    def __init__(self, channel=512, reduction=4, S=4):
        super(PSA, self).__init__()
        self.S = S
        self.convs = []
        for i in range(S):
            self.convs.append(layers.Conv2D(channel//S, kernel_size=2*(i+1)+1, padding='same'))

        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(Sequential([
                layers.GlobalAvgPool2D(keepdims=True),
                layers.Conv2D(channel//(S*reduction), kernel_size=1, use_bias=False),
                layers.Activation('relu'),
                layers.Conv2D(channel//S, kernel_size=1, use_bias=False),
                layers.Activation('sigmoid')
            ]))

        self.softmax = tf.nn.softmax

    def call(self, x):
        b, h, w, c = x.get_shape()

        # Step1: SPC module
        SPC_out = tf.reshape(x, shape=(b, h, w, self.S, c//self.S))  # bs, h, w, s, ci
        SPC_out_list = []
        for idx, conv in enumerate(self.convs):
            SPC_out_list.append(conv(SPC_out[:, :, :, idx, :]))

        SPC_out = tf.stack(SPC_out_list, axis=3)

        # Step2: SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append((se(SPC_out[:, :, :, idx, :])))
        SE_out = tf.stack(se_out, axis=3)
        SE_out = tf.broadcast_to(SE_out, SPC_out.get_shape())

        # Step3: Softmax
        softmax_out = self.softmax(SE_out)

        # Step4: SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = tf.reshape(PSA_out, shape=(b, h, w, -1))

        return PSA_out

if __name__ == '__main__':
    input = tf.random.normal((50, 7, 7, 512))
    psa = PSA(channel=512, reduction=8)
    output = psa(input)
    print(output.shape)
