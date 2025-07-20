import io
import re
import string
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import os
import torch
import os
from IPython import display
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, Input, TimeDistributed, GlobalAveragePooling2D, ConvLSTM2D, Reshape, Concatenate
from transformers import TFAutoModel
from tensorflow.keras.layers import Reshape, RepeatVector, Concatenate

# Load the pre-trained model using the Hugging Face interface
model_checkpoint = "facebook/wav2vec2-base"
huggingface_model = TFAutoModel.from_pretrained(model_checkpoint, trainable=False, from_pt=True)

# Define the inputs to the model
input_values = tf.keras.Input(shape=(16000,), dtype=tf.float32)

# Pass the inputs through the Wav2Vec model
wav2vec_output = huggingface_model(input_values)


# Create the TensorFlow functional API model
audio_model = tf.keras.Model(inputs=input_values, outputs=wav2vec_output)

# Print the model summary
audio_model.summary()

# Time model
# Define input layers
time_stamps = Input(shape=(50, 2))

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])

# Reshape the concatenated tensor
reshaped = Reshape((-1, 16))(concatenated)

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)

# Apply dense layer for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

# Define the model
time_model = Model(inputs=time_stamps, outputs=lstm_output, name='time_model')

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
time_model.summary()

# Reshape word_model output to match the shape of audio_model output
audio_model_output = layers.GlobalAveragePooling1D()(audio_model.output[1])
# Drop-out layer before the final Classification-Head
audio_model_output = layers.Dropout(0.5) (audio_model_output)

concatenated_output = Concatenate()([audio_model_output, time_model.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
combined= Dense(1, activation='sigmoid')(concatenated_output)
audio_time_model = Model(inputs=[audio_model.input, time_model.input], outputs=combined, name='audio_time_model')
audio_time_model.summary()

# Compile and train the model
audio_time_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
audio_time_model.fit([audio_train, time_train], y_train,
                       epochs=50, batch_size=16, validation_data=([audio_val, time_val], y_val),
                       callbacks=callback)

