import tensorflow as tf
from tensorflow.keras import layers


class Mlp(tf.keras.Model):
    def __init__(self, n_actions):
        super(Mlp, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(n_actions, activation='softmax')

    def call(self, obs):
        x = self.dense1(obs)
        return self.dense2(x)
