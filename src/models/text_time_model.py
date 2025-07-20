# Load pretrained word2vec embeddings
from gensim.models import KeyedVectors
word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors", mmap='r')

# Import libaries
import io
import re
import string
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras import layers
from xml.etree.ElementTree import parse
from xml.etree.ElementTree import fromstring
import jieba
import matplotlib.pyplot as plt
import random
import os
import torch

import pickle
aaaa# save dictionary to person_data.pkl file
with open('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/final_combined_data/vocab_dict.pkl', 'wb') as fp:
    pickle.dump(vocab, fp)
    print('dictionary saved successfully to file')

# Load the vocabulary dictionary
import pickle
with open('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original/vocab_dict.pkl', 'rb') as handle:
    data = handle.read()
vocab = pickle.loads(data)

vocab = tokenizer.word_index

# Obtain the word embeddings for each word 
from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
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

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, Input, TimeDistributed, GlobalAveragePooling2D, ConvLSTM2D, Reshape, Concatenate

# Create the embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                                output_dim=300,
                                weights=[embedding_vectors],
                                input_length=50,
                                trainable=False)

from transformers import AutoFeatureExtractor
#evaluate
from datasets import load_dataset
from datasets import load_from_disk
#final_combined_dataset=load_from_disk('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/final_combined_data')

# Load the dataset
from datasets import load_dataset
from datasets import load_from_disk
import numpy as np
final_combined_dataset=load_from_disk('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original')
final_combined_dataset=final_combined_dataset.train_test_split(test_size=0.2)
final_trained_dataset=final_combined_dataset['train'].train_test_split(test_size=0.2)
from tensorflow.keras.preprocessing.sequence import pad_sequences# Convert nested lists to numpy arrays
word_train = np.asarray(final_trained_dataset['train']['word']).astype(np.float32)
time_train = np.asarray(final_trained_dataset['train']['time_stamps']).astype(np.float32)
word_val = np.asarray(final_trained_dataset['test']['word']).astype(np.float32)
time_val = np.asarray(final_trained_dataset['test']['time_stamps']).astype(np.float32)
y_train=np.asarray(final_trained_dataset['train']['label']).astype(np.float32)
y_val=np.asarray(final_trained_dataset['test']['label']).astype(np.float32)
word_test=np.asarray(final_combined_dataset['test']['word']).astype(np.float32)
time_test=np.asarray(final_combined_dataset['test']['time_stamps']).astype(np.float32)
y_test=np.asarray(final_combined_dataset['test']['label']).astype(np.float32)

# Build the text and time model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate, Reshape, LSTM, Dense

# Define input layers
word_input = Input(shape=(50))
time_stamps = Input(shape=(50, 2))

# Define embedding layer

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])

# Reshape the concatenated tensor
#reshaped = Reshape((-1, 16))(concatenated)

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)

# Apply dense layer for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

# Define the model
model = Model(inputs=[word_input, time_stamps], outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# Print the model summary
model.summary()

# Train the model
import tensorflow as tf
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10)
model.fit([word_train, time_train], y_train, shuffle =True, validation_data=([word_val, time_val], y_val), epochs=50, batch_size=16, callbacks=[callback])

# Evaluate the model using the test set
model.evaluate([word_test, time_test], y_test)