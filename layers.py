import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class PositionLayer(tf.keras.layers.Layer):

   def __init__(self, embedding_size, **kwargs):
      self.embedding_size = embedding_size;
      super(PositionLayer, self).__init__(**kwargs)

   def build(self, input_shape):
      super(PositionLayer, self).build(input_shape)

   def call(self, x):

      mask = K.expand_dims(K.cast(K.arange(start =0, stop= K.shape(x)[1] +1), 'float32'), axis=-1);
      bins = K.expand_dims(K.cast(K.arange(self.embedding_size // 2) * 2, 'float32'), axis=0);

      evens = K.dot(mask, 1.0 / K.pow( 10000.0, bins / self.embedding_size));
      odds = tf.identity(evens);

      evens = K.sin(evens)[1:, :];
      odds = K.cos(odds)[1:, :];

      pos = K.reshape(K.stack([evens, odds], axis=2), (-1, K.shape(x)[1], self.embedding_size));
      y = K.expand_dims(x, axis=-1);

      return pos * y;

   def compute_output_shape(self, input_shape):
      return input_shape + (self.embedding_size,);

class MaskLayerLeft(tf.keras.layers.Layer):

   def __init__(self, **kwargs):
      super(MaskLayerLeft, self).__init__(**kwargs)

   def build(self, input_shape):
      super(MaskLayerLeft, self).build(input_shape)

   def call(self, x):

      length = K.shape(x)[1];
      rank = tf.ones(shape=(1, length), dtype='float32');
      y = K.expand_dims(x, axis=-1);

      mask = K.dot(y, rank);
      return tf.transpose(mask, (0,2,1));

   def compute_output_shape(self, input_shape):
      return input_shape + (input_shape[1]);

class MaskLayerRight(tf.keras.layers.Layer):

   def __init__(self, **kwargs):
      super(MaskLayerRight, self).__init__(**kwargs)

   def build(self, input_shape):
      super(MaskLayerRight, self).build(input_shape)

   def call(self, x):

      right = x[0];
      left = x[1];

      length = K.shape(right)[1];
      rank = tf.ones(shape=(1, length), dtype='float32');
      y = K.expand_dims(left, axis=-1);

      mask = K.dot(y, rank);
      return tf.transpose(mask, (0,2,1));

   def compute_output_shape(self, input_shape):
      return input_shape + (input_shape[1]);

class MaskLayerTriangular(tf.keras.layers.Layer):

   def __init__(self, **kwargs):
      super(MaskLayerTriangular, self).__init__(**kwargs)

   def build(self, input_shape):
      super(MaskLayerTriangular, self).build(input_shape)

   def call(self, x):

      t = tf.ones(shape= (K.shape(x)[0], K.shape(x)[1], K.shape(x)[1]));
      tri = tf.matrix_band_part(t, -1, 0);

      rank = tf.ones(shape=(1, K.shape(x)[1]), dtype='float32');
      y = K.expand_dims(x, axis=-1);

      mask = K.dot(y, rank);
      return tri * tf.transpose(mask, (0,2,1));

   def compute_output_shape(self, input_shape):
      return input_shape + (input_shape[1],);

class LayerNormalization(tf.keras.layers.Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
									 initializer= tf.keras.initializers.Ones(), trainable= self.trainable)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
									initializer= tf.keras.initializers.Zeros(), trainable= self.trainable)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape;

class SelfLayer(tf.keras.layers.Layer):

    def __init__(self, embedding_size, key_size, **kwargs):
        self.embedding_size = embedding_size;
        self.key_size = key_size;
        self.denom = math.sqrt(embedding_size);
        super(SelfLayer, self).__init__(**kwargs);

    def build(self, input_shape):

        self.K = self.add_weight( shape =(self.embedding_size, self.key_size),
                                name="K", trainable = True,
                                initializer = 'glorot_uniform');
        self.V = self.add_weight( shape =(self.embedding_size, self.key_size),
                                name="V", trainable = True,
                                initializer = 'glorot_uniform');
        self.Q = self.add_weight( shape =(self.embedding_size, self.key_size),
                                name="Q", trainable = True,
                                initializer = 'glorot_uniform');
        super(SelfLayer, self).build(input_shape);

    def call(self, inputs):

        Q = tf.tensordot(inputs[0], self.Q, axes = [[2], [0]]);
        K = tf.tensordot(inputs[1], self.K, axes = [[2], [0]]);
        V = tf.tensordot(inputs[2], self.V, axes = [[2], [0]]);

        A = tf.keras.backend.batch_dot(Q, tf.transpose(K, (0,2,1)));
        A = A / self.denom;

        A = tf.exp(A) * inputs[3];
        A = A / tf.reshape( tf.reduce_sum(A, axis = 2), (-1, tf.shape(inputs[0])[1] ,1));

        A = layers.Dropout(rate = 0.1) (A);
        return tf.keras.backend.batch_dot(A, V);

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.key_size);
