import tensorflow as tf
from lib.layers import Conv, ConvBnAct, FullyConnected, GlobalAvgPool, MaxPool


class BasicFE(tf.keras.layers.Layer):
    def __init__(
            self,
            name : str,
            M : int = 1,
    ):
        super(BasicFE, self).__init__(name=name)

        self.c1 = ConvBnAct(8 * M, 3, stride = 2, name='conv_1') # 128
        self.c2 = ConvBnAct(8 * M, 3, name='conv_2')
        self.p1 = MaxPool(name='pool_1') # 64
        self.c3 = ConvBnAct(16 * M, 3, name='conv_3')
        self.c4 = ConvBnAct(16 * M, 3, name='conv_4')
        self.p2 = MaxPool(name='pool_2') # 32
        self.c5 = ConvBnAct(32 * M, 3, name='conv_5')
        self.c6 = ConvBnAct(32 * M, 3, name='conv_6')
        self.p3 = MaxPool(name='pool_3') # 16
        self.c7 = ConvBnAct(64, 3, name='conv_7')
        self.c8 = ConvBnAct(64, 3, name='conv_8')

        self.global_pool = GlobalAvgPool(name='global_pool')

        self.fc_1 = FullyConnected(units=64)
        self.fc_2 = FullyConnected(units=32)
        self.fc_3 = FullyConnected(units=12)

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = inputs

        x = self.c1(x, training=training)
        x = self.c2(x, training=training)
        x = self.p1(x)
        x = self.c3(x, training=training)
        x = self.c4(x, training=training)
        x = self.p2(x)
        x = self.c5(x, training=training)
        x = self.c6(x, training=training)
        x = self.p3(x)
        x = self.c7(x, training=training)
        x = self.c8(x, training=training)
        x = self.global_pool(x)

        x = self.fc_1(x, training=training)
        x = self.fc_2(x, training=training)
        x = self.fc_3(x, training=training)

        return x
