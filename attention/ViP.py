import tensorflow as tf
from tensorflow.keras import layers


class MLP(layers.Layer):
    def __init__(self, hidden_features, out_features, drop=0.1):
        super(MLP, self).__init__()
        self.fc1 = layers.Dense(hidden_features, activation='gelu')
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        return self.drop(self.fc2(self.drop(self.fc1(x))))


class WeightedPermuteMLP(layers.Layer):
    def __init__(self, dim, seg_dim=8, qkv_bias=False, proj_drop=0.):
        super(WeightedPermuteMLP, self).__init__()
        self.seg_dim = seg_dim
        self.mlp_c = layers.Dense(dim, use_bias=qkv_bias)
        self.mlp_h = layers.Dense(dim, use_bias=qkv_bias)
        self.mlp_w = layers.Dense(dim, use_bias=qkv_bias)

        self.reweighting = MLP(dim // 4, dim * 3)

        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B, H, W, C = x.get_shape()

        c_embed = self.mlp_c(x)

        S = C // self.seg_dim
        h_embed = tf.reshape(tf.transpose(tf.reshape(x, (B, H, W, self.seg_dim, S)), (0, 3, 2, 1, 4)),
                             (B, self.seg_dim, W, H * S))
        h_embed = tf.reshape(tf.transpose(tf.reshape(self.mlp_h(h_embed), (B, self.seg_dim, W, H, S)), (0, 3, 2, 1, 4)),
                             (B, H, W, C))

        w_embed = tf.reshape(tf.transpose(tf.reshape(x, (B, H, W, self.seg_dim, S)), (0, 3, 2, 1, 4)),
                             (B, self.seg_dim, W, H * S))
        w_embed = tf.reshape(tf.transpose(tf.reshape(self.mlp_w(w_embed), (B, self.seg_dim, W, H, S)), (0, 3, 2, 1, 4)),
                             (B, H, W, C))

        weight = tf.reduce_mean(tf.reshape(tf.transpose((c_embed + h_embed + w_embed), (0, 3, 1, 2)), (B, C, -1)),
                                axis=2)
        weight = tf.expand_dims(tf.expand_dims(
            tf.nn.softmax(tf.transpose(tf.reshape(self.reweighting(weight), (B, C, 3)), (2, 0, 1)), axis=0), axis=2),
            axis=2)

        x = c_embed * weight[0] + w_embed * weight[1] + h_embed * weight[2]

        x = self.proj_drop(self.proj(x))

        return x


if __name__ == '__main__':
    input = tf.random.normal((64, 8, 8, 512))
    seg_dim = 8
    vip = WeightedPermuteMLP(512, seg_dim)
    output = vip(input)
    print(output.shape)
