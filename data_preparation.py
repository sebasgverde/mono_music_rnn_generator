# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
def prepare_data():
    # ### Download the Shakespeare dataset
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # The unique characters in the file
    vocab = sorted(set(text))

    import pdb; pdb.set_trace()

    # ## Process the text
    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])



    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text)//seq_length

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


    # The `batch` method lets us easily convert these individual characters to sequences of the desired size.
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    # For each sequence, duplicate and shift it to form the input and target text by using the `map` method to apply a simple function to each batch:
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # ### Create training batches
    #
    # We used `tf.data` to split the text into manageable sequences. But before feeding this data into the model, we need to shuffle the data and pack it into batches.
    # Batch size
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch//BATCH_SIZE

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    import pdb; pdb.set_trace()
    return dataset, len(vocab), BATCH_SIZE, idx2char, steps_per_epoch, char2idx
