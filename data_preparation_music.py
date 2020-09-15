# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import midi
import pickle
from midiutil.MidiFile import MIDIFile

DURATION_DIMENTION = 16
def note_tuple_2_index(min_note, note_tuple):
    pitch = note_tuple[0] - min_note
    duration = note_tuple[1]-1

    return DURATION_DIMENTION*pitch + duration

def index_2_note_tuple(min_note, index):
    duration = (index % DURATION_DIMENTION) + 1
    pitch = (int(index / DURATION_DIMENTION)) + min_note
    return [pitch, duration]

def sequenceVector2midiMelody(seqVector, file_dir):
    MyMIDI = MIDIFile(1)
    track = 0
    time = 0
    MyMIDI.addTrackName(track,time,"Sample Track")
    MyMIDI.addTempo(track,time,120)
    time = 0
    for tuple_note in seqVector:
        pitch = tuple_note[0]
        duration = tuple_note[1]/4
        if duration != 0:
            # MyMIDI.addNote(track,channel,pitch,time,duration,volume)
            MyMIDI.addNote(0,0,pitch,time,duration,100)
            time += duration

    binfile = open(file_dir, 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

    # #save also as pickle
    # pickle.dump(seqVector, open(file_dir + '.p','wb'))

def prepare_data(db_type, validation_split=0.3):
    """
    validation_split: rate of the data that is gona be use as validation (check loss
    in every timestep but without using it for training)

    Note: at the end i am using the validation dataset, but this split thing would be usefull
    if we don't have the data previusly splited
    """
    # ### Download the Shakespeare dataset
    if db_type == "control":
        tensor=pickle.load(open("data/train_song_list_cleaned.p"))
        tensor_validation=pickle.load(open("data/validation_song_list_cleaned.p"))
    elif db_type == "interval":
        tensor=pickle.load(open("data/train_song_list_intervals_cleaned.p"))
        tensor_validation=pickle.load(open("data/validation_song_list_intervals_cleaned.p"))
    elif db_type == "db12":
        tensor=pickle.load(open("data/train_song_list_db12_cleaned.p"))
        tensor_validation=pickle.load(open("data/validation_song_list_db12_cleaned.p"))
    # import pdb; pdb.set_trace()
    # pitches_list = []
    # for song in tensor:
    #     for elem in song:
    #         pitches_list.append(elem[0])
    pitches_list = [elem[0] for song in tensor for elem in song] + [elem[0] for song in tensor_validation for elem in song]
    min_note = min(pitches_list)
    max_note = max(pitches_list)
    resolution = 16

    # note_tuple_2_index = np.array((max_note-min_note, resolution))
    assert note_tuple_2_index(60, (60, 8)) == 7
    assert note_tuple_2_index(60, (61, 8)) == 23
    assert note_tuple_2_index(60, (61, 15)) == 30
    assert note_tuple_2_index(60, (62, 15)) == 46

    assert index_2_note_tuple(60, 7) == [60, 8]
    assert index_2_note_tuple(60, 23) == [61, 8]
    assert index_2_note_tuple(60, 30) == [61, 15]
    assert index_2_note_tuple(60, 46) == [62, 15]

    # esto podria ser diferente para intervals usando solo los que existen, pero revisando,
    # hay 71 y usando el rango min max son 86, francamente la diferencia no es tanta,
    # y asi el modelo queda como soportando todos los posibles intervalos y no solo los
    # presentes en la database, ademas checqueando, los algoritmos de note_tuple_2_index
    # y index_2_note_tuple funcionan igualmente para el casa intervals, asi que asi se quedara
    # text_as_int = [note_tuple_2_index(min_note, elem) for song in tensor for elem in song]

    # split_index = (int)(len(text_as_int)*(1-validation_split))

    # text_as_int_train = text_as_int[0:split_index]
    # text_as_int_validation = text_as_int[split_index:]
    text_as_int_train = [note_tuple_2_index(min_note, elem) for song in tensor for elem in song]
    text_as_int_validation = [note_tuple_2_index(min_note, elem) for song in tensor_validation for elem in song]
    # validation_data
    """
    asserts para casos de la database, que no haya canciones menores a 12,
    que no haya duraciones mayores a 16, etc
    """
    # min size of songs
    if db_type == "interval":
        assert min([len(elem) for elem in tensor]) == 11
        assert min([len(elem) for elem in tensor_validation]) == 11
    else:
        assert min([len(elem) for elem in tensor]) == 12
        assert min([len(elem) for elem in tensor_validation]) == 12

    # min duration 1 max duration 16
    assert min([elem[1] for song in tensor for elem in song]) == 1
    assert max([elem[1] for song in tensor for elem in song]) == 16
    assert min([elem[1] for song in tensor_validation for elem in song]) == 1
    assert max([elem[1] for song in tensor_validation for elem in song]) == 16
    # for song in tensor:

    # path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # Read, then decode for py2 compat.
    # text_as_int = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # The unique characters in the file


    # TODO
    # import pdb; pdb.set_trace()
    # creo que el problema en ese entrenamiento es por que hay en realidad 17 duraciones
    # y no 16, me toca revisar ese caso, o eliminar las duraciones 0 del dataset
    #
    # por ahora puse ese + 1 que me permite entrenar completo pero los resultados no me
    # convencen estas dimensiones habra que revisarlas bien con calma
    vocab_size = (max_note - min_note) * (DURATION_DIMENTION+1)

    # ## Process the text
    # Creating a mapping from unique characters to indices
    # char2idx = {u:i for i, u in enumerate(vocab)}
    # idx2char = np.array(vocab)

    # text_as_int = np.array([char2idx[c] for c in text])



    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text_as_int_train)//seq_length

    # Create training examples / targets
    char_dataset_train = tf.data.Dataset.from_tensor_slices(text_as_int_train)
    char_dataset_validation = tf.data.Dataset.from_tensor_slices(text_as_int_validation)

    # The `batch` method lets us easily convert these individual characters to sequences of the desired size.
    sequences_train = char_dataset_train.batch(seq_length+1, drop_remainder=True)
    sequences_validation = char_dataset_validation.batch(seq_length+1, drop_remainder=True)

    # For each sequence, duplicate and shift it to form the input and target text by using the `map` method to apply a simple function to each batch:
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset_train = sequences_train.map(split_input_target)
    dataset_validation = sequences_validation.map(split_input_target)

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

    dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset_validation = dataset_validation.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return dataset_train, dataset_validation, vocab_size, BATCH_SIZE, steps_per_epoch, min_note

# prepare_data()
# midi2sequenceVectorWithTime("/home/sebastian/aiva/code/db_manager/data/preset/rock/accompaniments/Rock Preset - 4_4 Accompaniments.mid")
