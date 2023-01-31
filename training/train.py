import os
import pathlib
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
# from IPython import display

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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

# PATH FOR WORKING REMOTELY ON MONTUS SERVER
# DATASET_PATH = "full_df.csv"
# DATA_PATH = "/home/lagarrueal/voice_commands/data/"

# PATH FOR WORKING LOCALLY
DATASET_PATH = "dataframes/full_df.csv"
DATA_PATH = "/home/lagarrueal/voice_commands/VoiceCommands/data/"


COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_background_noise_"]
TARGETS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "_background_noise_"]
map_class_to_id = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9, 'unknown':10, '_background_noise_':11}

dataset = pd.read_csv(DATASET_PATH)
print(f"{bcolors.OKBLUE}Dataset shape : {dataset.shape}{bcolors.ENDC}")
dataset = dataset.head(100)
AUTOTUNE = tf.data.AUTOTUNE

print(f"{bcolors.OKBLUE}Starting to decode audio files...{bcolors.ENDC}")
dataset['waveform'] = dataset.apply(lambda row: decode_audio(tf.io.read_file(DATA_PATH + row['label'] + "/" + row['filename'])), axis=1)
dataset = dataset.dropna()
dataset = dataset.reset_index(drop = True)
print(f"{bcolors.OKBLUE}Starting to convert audio files to spectrograms...{bcolors.ENDC}")
dataset['spectrogram'] = dataset['waveform'].apply(lambda x: get_spectrogram(x))
print(f"{bcolors.OKBLUE}Transforming labels{bcolors.ENDC}")
dataset['label'] = dataset['label'].apply(lambda x: x if x in COMMANDS else "unknown")
dataset['label'] = dataset['label'].apply(lambda x: map_class_to_id[x])


train_ds = dataset[dataset['set'] == 'training']
train_ds = train_ds.reset_index(drop = True)
val_ds = dataset[dataset['set'] == 'validation']
val_ds = val_ds.reset_index(drop = True)
test_ds = dataset[dataset['set'] == 'testing']
test_ds = test_ds.reset_index(drop = True)



for spectrogram in train_ds['spectrogram'][0:1]:
    input_shape = spectrogram.shape
print(f'{bcolors.OKBLUE}Input shape: {input_shape}{bcolors.ENDC}')
num_labels = len(TARGETS)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
# norm_layer.adapt(data = train_ds['spectrogram'])

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

print(f"{bcolors.WARNING}{model.summary()}{bcolors.ENDC}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

X_train, y_train = train_ds['spectrogram'], train_ds['label']
X_val, y_val = val_ds['spectrogram'], val_ds['label']
X_train_tensor, X_val_tensor = [], []

for i in range(len(X_train)):
	X_train_tensor.append(tf.convert_to_tensor(X_train[i], dtype=tf.float32))
for i in range(len(X_val)):
	X_val_tensor.append(tf.convert_to_tensor(X_val[i], dtype=tf.float32))

X_train_tensor = np.array([x.numpy() for x in X_train])
X_val_tensor = np.array([x.numpy() for x in X_val])

EPOCHS = 100
batch_size = 256
history = model.fit(
    x = X_train_tensor,
    y = y_train,
    epochs=EPOCHS,
    batch_size=batch_size,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
    validation_data = (X_val_tensor, y_val),
    verbose = 1
)

metrics = history.history


model.save('model_cnn.h5')

plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.savefig('loss_valoss.png')


plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

plt.savefig('training_curves.png')


test_audio = []
test_labels = []

for audio, label in zip(test_ds['spectrogram'], test_ds['label']):
    test_audio.append(tf.convert_to_tensor(audio, dtype=tf.float32))
    test_labels.append(label)

test_audio = np.array([x.numpy() for x in test_audio])
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'{bcolors.OKBLUE}Test set accuracy: {test_acc:.0%}{bcolors.ENDC}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=TARGETS,
            yticklabels=TARGETS,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.savefig('matconf.png')

# Convert the true and predicted labels to tensor format
y_true = tf.constant(y_true, dtype=tf.int64)
y_pred = tf.constant(y_pred, dtype=tf.int64)

# Calculate precision
precision = tf.metrics.Precision()
precision.update_state(y_true, y_pred)
precision = precision.result().numpy()

# Calculate recall
recall = tf.metrics.Recall()
recall.update_state(y_true, y_pred)
recall = recall.result().numpy()

# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

metrics = {"accuracy" : str(test_acc),
	        "precision" : str(precision),
            "recall" : str(recall),
	        "f1_score" : str(f1_score)}

# Writing to sample.json
with open("metrics.json", "w") as outfile:
    json.dump(metrics, outfile)

sample_file = DATA_PATH + "no/" + "eb67fcbc_nohash_2.wav"
label = 'no'
sample_data = get_spectrogram(decode_audio(tf.io.read_file(sample_file)))
sample_data_tensor = tf.convert_to_tensor(sample_data)
sample_data_tensor = tf.expand_dims(sample_data_tensor, 0)
sample_data_tensor = np.array(sample_data_tensor.numpy())
prediction = model.predict(sample_data_tensor)
plt.figure(figsize=(12, 8))
plt.bar(TARGETS, tf.nn.softmax(prediction[0]))
plt.title(f'Predictions for No')
plt.savefig('inference.png')
