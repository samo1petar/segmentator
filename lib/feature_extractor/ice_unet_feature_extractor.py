import tensorflow as tf
from lib.layers import Conv, ConvBnAct, DeConvBnAct, IceBlock, MaxPool


class IceUNetFE(tf.keras.layers.Layer):
    def __init__(
            self,
            name : str,
    ):
        super(IceUNetFE, self).__init__(name=name)

        self.cd1 = ConvBnAct(32, 3, stride=2, name='conv_down_1')
        self.cd2 = ConvBnAct(32, 3, name='conv_down_2')
        # self.p1 = MaxPool(name='pool_1')
        # self.cd3 = ConvBnAct(64, 3, name='conv_down_3')

        self.ice1 = IceBlock(16, 32, name='ice_down_1')
        # self.ice2 = IceBlock(16, 32, name='ice_down_2')

        self.p2 = MaxPool(name='pool_2')
        self.cd4 = ConvBnAct(64, 3, name='conv_down_4')

        # self.ice3 = IceBlock(32, 64, name='ice_down_3')
        self.ice4 = IceBlock(32, 64, name='ice_down_4')

        self.p3 = MaxPool(name='pool_3')
        self.cd5 = ConvBnAct(128, 3, name='conv_down_5')

        # self.ice5 = IceBlock(64, 128, name='ice_down_5')
        self.ice6 = IceBlock(64, 128, name='ice_down_6')

        self.p4 = MaxPool(name='pool_4')
        self.cd6 = ConvBnAct(256, 3, name='conv_down_6')

        # self.ice7 = IceBlock(64, 256, name='ice_down_7')
        self.ice8 = IceBlock(64, 256, name='ice_down_8')

        # self.p5 = MaxPool(name='pool_5')
        # self.cd7 = ConvBnAct(384, 3, name='conv_down_7')
        #
        # # self.ice9  = IceBlock(96, 384, name='ice_down_9')
        # self.ice10 = IceBlock(96, 384, name='ice_down_10')
        #
        # self.cu1 = ConvBnAct(256, 3, name='conv_up_1')
        # self.ice11 = IceBlock(64, 256, name='ice_up_1')

        self.cu2 = ConvBnAct(128, 3, name='conv_up_2')
        self.ice12 = IceBlock(64, 128, name='ice_up_2')

        self.cu3 = ConvBnAct(64, 3, name='conv_up_3')
        self.ice13 = IceBlock(32, 64, name='ice_up_3')

        self.cu4 = ConvBnAct(32, 3, name='conv_up_4')
        self.ice14 = IceBlock(16, 32, name='ice_up_4')

        self.cu5 = ConvBnAct(16, 3, name='conv_up_5')
        self.cu6 = ConvBnAct(16, 3, name='conv_up_6')

        self.upsample = tf.keras.layers.UpSampling2D()
        self.add = tf.keras.layers.Add()

        self.cu_out = Conv(1, 5, name='conv_out')

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = inputs
        # down
        x  = self.cd1(x, training=training)
        x  = self.cd2(x, training=training)
        # x  = self.p1(x)
        # x  = self.cd3(x, training=training)
        x2 = self.ice1(x, training=training)
        # x2 = self.ice2(x, training=training)
        x  = self.p2(x2)
        x  = self.cd4(x, training=training)
        # x  = self.ice3(x, training=training)
        x3 = self.ice4(x, training=training)
        x  = self.p3(x3)
        x  = self.cd5(x, training=training)
        # x  = self.ice5(x, training=training)
        x4 = self.ice6(x, training=training)
        x  = self.p4(x4)
        x  = self.cd6(x, training=training)
        # x  = self.ice7(x, training=training)
        x  = self.ice8(x, training=training)

        # removing last downscale
        # x  = self.p5(x5)
        # x  = self.cd7(x, training=training)
        # # x  = self.ice9(x, training=training)
        # x  = self.ice10(x, training=training)
        # # up
        # x  = self.upsample(x)
        # x  = self.cu1(x, training=training)
        # x  = self.add([x, x5])
        # x  = self.ice11(x, training=training)

        x  = self.upsample(x)
        x  = self.cu2(x, training=training)
        x  = self.add([x, x4])
        x  = self.ice12(x, training=training)

        x  = self.upsample(x)
        x  = self.cu3(x, training=training)
        x  = self.add([x, x3])
        x  = self.ice13(x, training=training)

        x  = self.upsample(x)
        x  = self.cu4(x, training=training)
        x  = self.add([x, x2])
        x  = self.ice14(x, training=training)

        x  = self.upsample(x)
        x  = self.cu5(x, training=training)
        x  = self.cu6(x, training=training)

        x  = self.cu_out(x, training=training)

        return x
