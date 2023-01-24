import os
import pathlib
import pandas as pd

# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
# from IPython import display


def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, 
    # normalized to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    try :
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
    except : 
        return None
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# DATASET_PATH = "full_df.csv"
DATASET_PATH = "dataframes/full_df.csv"
# DATA_PATH = "/home/lagarrueal/voice_commands/data/"
DATA_PATH = "/home/lagarrueal/voice_commands/VoiceCommands/data/"
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_background_noise_"]
TARGETS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "_background_noise_"]

dataset = pd.read_csv(DATASET_PATH)
print(dataset.shape)
dataset = dataset.head(100)
AUTOTUNE = tf.data.AUTOTUNE

print("Starting to decode audio files...")
dataset['waveform'] = dataset.apply(lambda row: decode_audio(tf.io.read_file(DATA_PATH + row['label'] + "/" + row['filename'])), axis=1)
dataset = dataset.dropna()
dataset = dataset.reset_index(drop = True)
print("Starting to convert audio files to spectrograms...")
dataset['spectrogram'] = dataset['waveform'].apply(lambda x: get_spectrogram(x))
print("Transforming labels")
dataset['label'] = dataset['label'].apply(lambda x: x if x in COMMANDS else "unknown")

batch_size = 128
train_ds = dataset[dataset['set'] == 'training']
train_ds = train_ds.reset_index(drop = True)
val_ds = dataset[dataset['set'] == 'validation']
val_ds = val_ds.reset_index(drop = True)
test_ds = dataset[dataset['set'] == 'testing']
test_ds = test_ds.reset_index(drop = True)



for spectrogram in train_ds['spectrogram'][0:1]:
    input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(TARGETS)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data = train_ds['spectrogram'])

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history

test_audio = []
test_labels = []

for audio, label in zip(test_ds['waveform'], test_ds['label']):
    test_audio.append(audio.numpy())
    test_labels.append(label)

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

model.save('model_cnn.h5')
