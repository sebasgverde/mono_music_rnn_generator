# Next define a function to build the model.
#
# Use `CuDNNGRU` if running on GPU.

# In[ ]:
import tensorflow as tf

# if tf.test.is_gpu_available():
  # rnn = tf.keras.layers.CuDNNGRU
  # rnn = tf.keras.layers.CuDNNLSTM

  # # it's supposed to be the new version replacing the others
  # rnn = tf.compat.v1.keras.layers.CuDNNGRU

# else:
#   import functools
#   rnn = functools.partial(
#     tf.keras.layers.GRU, recurrent_activation='sigmoid')
    # tf.keras.layers.LSTM, recurrent_activation='sigmoid')
    # tf.contrib.rnn.UGRNNCell, recurrent_activation='sigmoid') # doesn't work

# In[ ]:

def define_rnn_layer(rnn_type):
    if tf.test.is_gpu_available():
        if rnn_type == "gru":
            rnn = tf.keras.layers.CuDNNGRU
        elif rnn_type == "lstm":
            rnn = tf.keras.layers.CuDNNLSTM

    return rnn

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, rnn_type):
  rnn = define_rnn_layer(rnn_type)
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model
