import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import FeedForwardPolicy

def custom_cnn(scaled_images, **kwargs):
   activ = tf.nn.leaky_relu
   layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
   layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
   layer_2 = conv_to_fc(layer_2)
   return activ(linear(layer_2, 'fc1', n_hidden=64, init_scale=np.sqrt(2)))