import data_preparation_music as data_preparation
import pickle
import tensorflow as tf
tf.enable_eager_execution()

# import numpy as np

import nn_model

import argparse

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--db_type', type=str)
parser.add_argument('--rnn_type', type=str)
parser.add_argument('--rnn_units', type=str)
parser.add_argument('--checkpoints_folder', type=str)
parser.add_argument('--num_notes', type=str)
parser.add_argument('--output_uri', type=str)
parser.add_argument('--num_songs', type=int, default=1)
args = parser.parse_args()

# ## Generate text

# ### Restore the latest checkpoint

# To keep this prediction step simple, use a batch size of 1.
# Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.
# To run the model with a different `batch_size`, we need to rebuild the model and restore the weights from the checkpoint.

# rnn_type = "gru"
# checkpoint_dir = "./training_checkpoints_db12_" + rnn_type + "_units_512"

# db_type = "control"
# # db_type = "interval"
# # db_type = "db12"
#
# rnn_type = "lstm"
# # rnn_type = "gru"
#
# units = "1024"

db_type = args.db_type
rnn_type = args.rnn_type
units = args.rnn_units
checkpoints_folder = args.checkpoints_folder

# checkpoint_dir = "./layers_exp_2/selected/training_checkpoints_"+db_type+"_" + rnn_type + "_units_"+units
checkpoint_dir = checkpoints_folder+"/training_checkpoints_"+db_type+"_" + rnn_type + "_units_"+units

model_params = pickle.load(open(checkpoint_dir + "/model_params.p"))

vocab_size = model_params["vocab_size"]
embedding_dim = model_params["embedding_dim"]
rnn_units = model_params["rnn_units"]
min_note = model_params["min_note"]

# import pdb; pdb.set_trace()
tf.train.latest_checkpoint(checkpoint_dir)

model = nn_model.build_model(vocab_size, embedding_dim, rnn_units, batch_size=1, rnn_type=rnn_type)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()


# ### The prediction loop
#
# The following code block generates the text:
# * It starts by choosing a start string, initializing the RNN state and setting the number of characters to generate.
# * Get the prediction distribution of the next character using the start string and the RNN state.
# * Use a multinomial distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model.
# * The RNN state returned by the model is fed back into the model so that it now has more context, instead of only one word. After predicting the next word, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted words.

def generate_text(model, start_string, num_notes):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = num_notes

    # Converting our start string to numbers (vectorizing)
    input_eval = [data_preparation.note_tuple_2_index(min_note,tuple_elem) for tuple_elem in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(data_preparation.index_2_note_tuple(min_note,predicted_id))

    start_string.extend(text_generated)
    return start_string

for j in range(args.num_songs):
    #interval
    if args.db_type == "interval":
        initial_pitch = 62
        default_last_duration = 4
        interval_song = generate_text(model, start_string=[[2,8],[1,4]], num_notes=int(args.num_notes))

        song = []
        current_pitch = initial_pitch
        for i in range(len(interval_song)):
            inter_tuple = interval_song[i]
            song.append([current_pitch, inter_tuple[1]])
            current_pitch += inter_tuple[0]
            i += 1
        else:
            song.append([current_pitch, default_last_duration])
    else:
        #control and db12
        song = generate_text(model, start_string=[[62,8],[64,4],[65,4]], num_notes=int(args.num_notes))



    pickle.dump(song, open(args.output_uri+"_"+str(j)+".p", "wb"))
    data_preparation.sequenceVector2midiMelody(song, args.output_uri+"_"+str(j)+".mid")
