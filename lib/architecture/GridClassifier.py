import tensorflow as tf
from lib.feature_extractor.classification_unet_feature_extractor import CLS_UNetFE
from lib.layers import FullyConnected, GlobalAvgPool


class Model(tf.keras.Model):
    def __init__(
            self,
            name : str,
            M    : int = 1,
    ):
        super(Model, self).__init__(name=name)

        self.feature_extractor = CLS_UNetFE(name='feature_extractor', M=M)

        self.global_pool = GlobalAvgPool(name='global_pool')

        self.fc_1 = FullyConnected(units=64)
        self.fc_2 = FullyConnected(units=32)
        self.fc_3 = FullyConnected(units=12)

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = self.feature_extractor(inputs, training=training)

        x = self.global_pool(x)

        x = self.fc_1(x, training=training)
        x = self.fc_2(x, training=training)
        x = self.fc_3(x, training=training)

        return x
