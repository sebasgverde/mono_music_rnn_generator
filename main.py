
# coding: utf-8

"""
based in the notebook in https://www.tensorflow.org/tutorials/sequences/text_generation
"""
# ##### Copyright 2018 The TensorFlow Authors.

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ## Setup

# ### Import TensorFlow and other libraries

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# # This run the model step by step, use only for debugging
# tf.enable_eager_execution()

import numpy as np
import os
import time


import data_preparation_music as data_preparation
import nn_model

import argparse

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--db_type', type=str)
parser.add_argument('--rnn_type', type=str)
parser.add_argument('--rnn_units', type=int)
parser.add_argument('--epochs', type=int)
args = parser.parse_args()


# db_type = "control"
# db_type = "interval"
# db_type = "db12"

# rnn_type = "lstm"
# rnn_type = "gru"

# Number of RNN units
# rnn_units = 256
# rnn_units = 512
# rnn_units = 1024

# epochs=30

db_type = args.db_type
rnn_type = args.rnn_type
rnn_units = args.rnn_units
epochs = args.epochs

dataset_train, dataset_validation, vocab_size, BATCH_SIZE, steps_per_epoch, min_note = data_preparation.prepare_data(db_type)
# ## Build The Model



# Length of the vocabulary in chars
# vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256



model = nn_model.build_model(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE,
  rnn_type=rnn_type)

model.summary()

# ## Train the model

# At this point the problem can be treated as a standard classification problem. Given the previous RNN state, and the input this time step, predict the class of the next character.
# ### Attach an optimizer, and a loss function
# The standard `tf.keras.losses.sparse_categorical_crossentropy` loss function works in this case because it is applied across the last dimension of the predictions.
# Because our model returns logits, we need to set the `from_logits` flag.
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Configure the training procedure using the `tf.keras.Model.compile` method. We'll use `tf.train.AdamOptimizer` with default arguments and the loss function.
model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)


# ### Configure checkpoints


# Directory where the checkpoints will be saved
checkpoint_dir = './models/training_checkpoints_' + db_type + '_'+ rnn_type +'_units_' + str(rnn_units)
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_freq=int(steps_per_epoch*epochs/5)) # to save only last one, can be every epoch or something like

checkpoint_min_train_loss=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + "/ckpt_min_train_loss",
    save_weights_only=True,
    save_best_only=True,
    mode="min",
    monitor="loss")

checkpoint_min_val_loss=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + "/ckpt_min_val_loss",
    save_weights_only=True,
    save_best_only=True,
    mode="min",
    monitor="val_loss")
# ### Execute the training

history = model.fit(dataset_train.repeat(), validation_data=dataset_validation, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback, checkpoint_min_train_loss, checkpoint_min_val_loss], verbose=2)
# history = model.fit(dataset.repeat(), epochs=10, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback], verbose=2)

import pickle
model_params = {"vocab_size": vocab_size, "embedding_dim": embedding_dim, "rnn_units": rnn_units, "min_note": min_note}
pickle.dump(model_params,open(checkpoint_dir + "/model_params.p", "wb"))
pickle.dump({"epochs": history.epoch, "training_loss": history.history["loss"], "validation_loss": history.history["val_loss"]},open(checkpoint_dir + "/learning_curve_info.p", "wb"))

import matplotlib.pyplot as plt
plt.plot(history.epoch, history.history["loss"], linewidth=2.0, label="training")
plt.plot(history.epoch, history.history["val_loss"], linewidth=2.0, label="validation")
plt.legend()
# plt.title(r'loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Cross Entropy)')
# plt.show()
plt.savefig(checkpoint_dir + '/learnig_curve.png')

# tambien lo guardo en caso de que llegue a necesitar informacion de aca, ya buscare como cargar el pickle sin errores
pickle.dump(history.history,open(checkpoint_dir + "/history.p", "wb"))

# # Advanced Trainig
#
# model = nn_model.build_model(
#   vocab_size = len(vocab),
#   embedding_dim=embedding_dim,
#   rnn_units=rnn_units,
#   batch_size=BATCH_SIZE)
#
# optimizer = tf.train.AdamOptimizer()
#
# # Training step
# epochs = 1
#
# for epoch in range(epochs):
#     start = time.time()
#
#     # initializing the hidden state at the start of every epoch
#     # initially hidden is None
#     hidden = model.reset_states()
#
#     for (batch_n, (inp, target)) in enumerate(dataset):
#           with tf.GradientTape() as tape:
#               # feeding the hidden state back into the model
#               # This is the interesting step
#               predictions = model(inp)
#               loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)
#
#           grads = tape.gradient(loss, model.trainable_variables)
#           optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#           if batch_n % 100 == 0:
#               template = 'Epoch {} Batch {} Loss {:.4f}'
#               print(template.format(epoch+1, batch_n, loss))
#
#     # saving (checkpoint) the model every 5 epochs
#     if (epoch + 1) % 5 == 0:
#       model.save_weights(checkpoint_prefix.format(epoch=epoch))
#
#     print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
#     print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
#
# model.save_weights(checkpoint_prefix.format(epoch=epoch))
