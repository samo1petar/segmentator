import tensorflow as tf
from lib.feature_extractor.basic_feature_extractor import BasicFE
from lib.feature_extractor.ice_unet_feature_extractor import IceUNetFE
from lib.feature_extractor.unet_feature_extractor import UNetFE


class Model(tf.keras.Model):
    def __init__(
            self,
            name : str,
            M    : int = 1,
    ):
        super(Model, self).__init__(name=name)

        self.feature_extractor = UNetFE(name='feature_extractor', M=M)

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = self.feature_extractor(inputs, training=training)

        x = tf.nn.sigmoid(x)

        return x
