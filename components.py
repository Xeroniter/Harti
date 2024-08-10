import gdown

import tensorflow as tf

import tensorflow.keras.utils as utils

inload = lambda url : gdown.download(url=url)

def imaloader(iurl, inputer):
  images = tf.io.decode_image(iurl, channels=3)
  images = tf.image.resize(images, inputer)
  images = tf.expand_dims(images, axis=0)
  images = tf.cast(images / 255.0, tf.float32)
  return images
