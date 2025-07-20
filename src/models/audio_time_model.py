# DONE

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
import pickle
from IPython import display
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, Input, TimeDistributed, GlobalAveragePooling2D, ConvLSTM2D, Reshape, Concatenate
from transformers import TFAutoModel
from wav2vec_feature_extractor import Wav2VecFeatureExtractor

model_checkpoint = "facebook/wav2vec2-base"
# huggingface_model = TFAutoModel.from_pretrained(model_checkpoint, trainable=False, from_pt=True)
# Pass the inputs through the Wav2Vec model
# wav2vec_output = huggingface_model(input_values)
# BEFORE WE HAD THIS INSTEAD
# Define the inputs to the model
input_values = tf.keras.Input(shape=(16000,), dtype=tf.float32)
audio_features = Wav2VecFeatureExtractor(model_checkpoint)(input_values)
audio_output = layers.GlobalAveragePooling1D()(audio_features)
audio_output = layers.Dropout(0.5)(audio_output)
audio_model = tf.keras.Model(inputs=input_values, outputs=audio_output, name='audio_model')
# Print the model summary
audio_model.summary()

# Dummy vocab and embedding setup
vocab_size = 1000
embedding_dim = 300
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=50, trainable=False)

# Word and time inputs
word_input = Input(shape=(50,), name='word_input')
time_stamps = Input(shape=(50, 2), name='time_input')

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)

# Word-time model
time_model = tf.keras.Model(inputs=[word_input, time_stamps], outputs=lstm_output, name='time_model')

# Print the model summary
time_model.summary()

concatenated_output = Concatenate()([audio_model.output, time_model.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
combined = Dense(1, activation='sigmoid')(concatenated_output)
audio_time_model = tf.keras.Model(inputs=[audio_model.input, time_model.input], outputs=combined, name='audio_time_model')
audio_time_model.summary()

# Compile and train the model
audio_time_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)

# Dummy data
audio_train = np.random.randn(10, 16000).astype(np.float32)
word_train = np.random.randint(0, vocab_size, size=(10, 50))
time_train = np.random.randn(10, 50, 2).astype(np.float32)
y_train = np.random.randint(0, 2, size=(10, 1))

audio_val = np.random.randn(2, 16000).astype(np.float32)
word_val = np.random.randint(0, vocab_size, size=(2, 50))
time_val = np.random.randn(2, 50, 2).astype(np.float32)
y_val = np.random.randint(0, 2, size=(2, 1))

# Train
audio_time_model.fit([audio_train, [word_train, time_train]], y_train,
                   epochs=50, batch_size=16,
 validation_data=([audio_val, [word_val, time_val]], y_val),
                   callbacks=[callback])

# Evaluate
audio_time_model.evaluate([audio_val, [word_val, time_val]], y_val)

# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model to the folder
audio_time_model.save(os.path.join("models", "audio_word_time_model.keras"))
