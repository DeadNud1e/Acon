from keras.layers import Layer
import tensorflow as tf


class AconC(Layer):
    r""" Tensorflow implementation of ACON activation (activate or not, <https://arxiv.org/pdf/2009.04759.pdf>).
        AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
        """
    def __init__(self, **kwargs):
        super(AconC, self).__init__(**kwargs)

    def build(self, input_shape):
        self.p1 = self.add_weight(name='p1',
                                shape=[1],
                                initializer='random_normal',
                                trainable=True)

        self.p2 = self.add_weight(name='p2',
                                shape=[1],
                                initializer='random_normal',
                                trainable=True)

        self.beta = self.add_weight(name='beta',
                                shape=[1],
                                initializer='random_normal',
                                trainable=True)

        super(AconC, self).build(input_shape)

    def call(self, x):
        return (self.p1 * x - self.p2 * x) * tf.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x
