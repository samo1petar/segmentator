import tensorflow as tf

# TODO -> rewrite for detection

class SoftmaxCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, multiplayer: float = 1, name: str = 'softmax_cross_entropy', **kwargs):
        super(SoftmaxCrossEntropy, self).__init__(name=name, **kwargs)
        self._multiplayer = multiplayer

    def call(self, labels: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:

        labels = tf.cast(labels, tf.float16)
        prediction = tf.cast(prediction, tf.float16)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=prediction,
        ))
        return loss * self._multiplayer
