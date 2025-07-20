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
from gensim.models.keyedvectors import Word2VecKeyedVectors
import tensorflow_hub as hub
from transformers import TFAutoModel
from keras.models import Sequential
import word2vec as vectors
from gensim.models import KeyedVectors
import pickle
from tensorflow.keras.layers import Reshape, RepeatVector, Concatenate, Input, Embedding, Concatenate, LSTM, Dense
from tensorflow.keras.models import Model

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

word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors", mmap='r')

# Save the vocabulary dictionary
with open('/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original_augmented/vocab_dict.pkl', 'wb') as fp:
    pickle.dump(vocab, fp)
    print('dictionary saved successfully to file')

# Import the vocabulary dictionary
with open('/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original_augmented/vocab_dict.pkl', 'rb') as handle:
    data = handle.read()
vocab = pickle.loads(data)

vocab = tokenizer.word_index

# Get the word embeddings for each word
#vocab = tokenizer.word_index

def get_weight_matrix():
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((len(vocab)+1, word2vec_vectors.vector_size))
    i=0
    for key in vocab.keys():
      if key=='OOV':
        continue
      elif key not in word2vec_vectors:
        i=i+1
        continue
      else:
        weight_matrix[i + 1] = word2vec_vectors[key]
        i=i+1
    return weight_matrix

embedding_vectors = get_weight_matrix()

embedding_layer = Embedding(input_dim=len(vocab) + 1,
                                output_dim=300,
                                weights=[embedding_vectors],
                                input_length=50,
                                trainable=False)

# Text and time model
# Define input layers
word_input = Input(shape=(50))
time_stamps = Input(shape=(50, 2))

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])

# Reshape the concatenated tensor
#reshaped = Reshape((-1, 16))(concatenated)

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)

# Apply dense layer for binary classification
#output = Dense(1, activation='sigmoid')(lstm_output)

# Define the model
word_model = Model(inputs=[word_input, time_stamps], outputs=lstm_output, name='word_model')

# Compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
word_model.summary()

# Reshape word_model output to match the shape of audio_model output
audio_model_output = layers.GlobalAveragePooling1D()(audio_model.output[1])
# Drop-out layer before the final Classification-Head
audio_model_output = layers.Dropout(0.5) (audio_model_output)

concatenated_output = Concatenate()([audio_model_output, word_model.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
combined= Dense(1, activation='sigmoid')(concatenated_output)
combined_model = Model(inputs=[audio_model.input, word_model.input], outputs=combined, name='combined_model')
combined_model.summary()

# Compile and train the model
combined_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
combined_model.fit([audio_train, combined_train], y_train,
                       epochs=50, batch_size=16, validation_data=([audio_val, combined_val], y_val),
                       callbacks=callback)

# Evaluate the model using the test set
combined_model.evaluate([audio_test, combined_test], y_test)

combined_model.save('/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/trained_models_shorts_augmented/audio_word_time')