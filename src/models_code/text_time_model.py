import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense
from tensorflow.keras.models import Model
import os
from weights import Weights
from utils import load_split
import pickle


_, word_train, time_train, y_train = load_split("pitt_split\train", load_audio=False)
_, word_val, time_val, y_val = load_split("pitt_split/val", load_audio=False)
_, word_test, time_test, y_test = load_split("pitt_split/test", load_audio=False)

with open(os.path.join("pitt_split", "vocab.pkl"), "rb") as f:
    data = f.read()
vocab = pickle.loads(data)

with open("embeddings/word2vec_vectors.pkl", "rb") as f:
    word2vec_vectors = pickle.load(f)

weight = Weights(vocab, word2vec_vectors)
embedding_vectors = weight.get_weight_matrix()

# Create embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=300,
                            weights=[embedding_vectors],
                            input_length=50,
                            trainable=False)

# Define inputs
word_input = Input(shape=(50,), name='word_input')

# # Define input layers
time_stamps = Input(shape=(50, 2), name='time_input')

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])

# LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)

# Apply dense layer for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

# Define the model
model = Model(inputs=[word_input, time_stamps], outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model.summary()

# Train with EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit([word_train, time_train], y_train,
          validation_data=([word_val, time_val], y_val),
          epochs=50,
          batch_size=16,
          shuffle=True,
          callbacks=[callback])

# Evaluate on test data
model.evaluate([word_test, time_test], y_test)

# Save model locally
os.makedirs("models", exist_ok=True)
# Save the model to the folder
model.save(os.path.join("models", "word_time_model.keras"))