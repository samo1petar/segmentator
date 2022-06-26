import tensorflow as tf
from lib.layers.Activation import Activation


class IceBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            input_filters: int,
            output_filters: int,
            kernel: int = 3,
            merge: str = 'add',
            activation: str = 'relu',
            **kwargs,
    ):
        super(IceBlock, self).__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel = kernel
        self.stride = 1 # emphasize
        self.activation_name = activation
        self.merge_name = merge

    def build(self, shape):
        self.conv1 = tf.keras.layers.Conv2D(
            filters     = self.input_filters,
            kernel_size = self.kernel,
            strides     = self.stride,
            padding     = 'same',
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters     = self.input_filters,
            kernel_size = self.kernel,
            strides     = self.stride,
            padding     = 'same',
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters     = self.output_filters,
            kernel_size = self.kernel,
            strides     = self.stride,
            padding     = 'same',
        )

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

        self.activation1 = Activation(self.activation_name)
        self.activation2 = Activation(self.activation_name)
        self.activation3 = Activation(self.activation_name)

        if self.merge_name == 'add':
            self.merge = tf.keras.layers.Add()
        elif self.merge_name == 'concat':
            self.merge = tf.keras.layers.Concatenate(axis=3)

    def call(self, inputs, training: bool = False):

        x = self.conv1(inputs, training=training)
        x = self.batch_norm1(x, training=training)
        x = self.activation1(x)

        x = self.conv2(x, training=training)
        x = self.batch_norm2(x, training=training)
        x = self.activation2(x)

        x = self.conv3(x, training=training)
        x = self.batch_norm3(x, training=training)
        x = self.activation3(x)

        x = self.merge([x, inputs])

        return x
