import tensorflow as tf
from lib.layers import Conv, ConvBnAct, DeConvBnAct, GlobalAvgPool, MaxPool


class CLS_UNetFE(tf.keras.layers.Layer):
    def __init__(
            self,
            name : str,
            M    : int = 1,
    ):
        super(CLS_UNetFE, self).__init__(name=name)

        self.cd1 = ConvBnAct(8 * M, 3, stride = 2, name='conv_down_1')
        self.cd2 = ConvBnAct(8 * M, 3, name='conv_down_2')
        self.p1 = MaxPool(name='pool_1')
        self.cd3 = ConvBnAct(16 * M, 3, name='conv_down_3')
        self.cd4 = ConvBnAct(16 * M, 3, name='conv_down_4')
        self.p2 = MaxPool(name='pool_2')
        self.cd5 = ConvBnAct(32 * M, 3, name='conv_down_5')
        self.cd6 = ConvBnAct(32 * M, 3, name='conv_down_6')
        self.p3 = MaxPool(name='pool_3')
        self.cd7 = ConvBnAct(64, 3, name='conv_down_7')
        self.cd8 = ConvBnAct(64, 3, name='conv_down_8')

        self.dec1 = DeConvBnAct(64, 3, name='deconv_1')
        self.cu1 = ConvBnAct(64 * M, 3, name='conv_up_1')
        self.cu2 = ConvBnAct(64 * M, 3, name='conv_up_2')
        self.dec2 = DeConvBnAct(64, 3, name='deconv_2')
        self.cu3 = ConvBnAct(64 * M, 3, name='conv_up_3')
        self.cu4 = ConvBnAct(64 * M, 3, name='conv_up_4')
        self.dec3 = DeConvBnAct(64, 3, name='deconv_3')
        self.cu5 = ConvBnAct(64 * M, 3, name='conv_up_5')
        self.cu6 = ConvBnAct(64 * M, 3, name='conv_up_6')
        self.dec4 = DeConvBnAct(64, 3, name='deconv_1')

        self.cu_out = Conv(64, 5, name='conv_out') # TODO add more layers

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = inputs

        # TODO add reconnecting connections

        x = self.cd1(x, training=training)
        x = self.cd2(x, training=training)
        x = self.p1(x)
        x = self.cd3(x, training=training)
        x = self.cd4(x, training=training)
        x = self.p2(x)
        x = self.cd5(x, training=training)
        x = self.cd6(x, training=training)
        x = self.p3(x)
        x = self.cd7(x, training=training)
        x = self.cd8(x, training=training)

        # x = self.dec1(x, training=training)
        # x = self.upsample(x)
        x = self.cu1(x, training=training)
        x = self.cu2(x, training=training)
        # x = self.dec2(x, training=training)
        # x = self.upsample(x)
        x = self.cu3(x, training=training)
        x = self.cu4(x, training=training)
        # x = self.dec3(x, training=training)
        # x = self.upsample(x)
        x = self.cu5(x, training=training)
        x = self.cu6(x, training=training)
        # x = self.dec4(x, training=training)
        x = self.cu_out(x, training=training)

        return x
