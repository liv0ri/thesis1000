import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense
from tensorflow.keras.models import Model
from weights import Weights
from utils import load_split, pad_sequences_and_times_np
import pickle
from config import TRAIN_PATH, TEST_PATH, VAL_PATH

# Define a constant for the maximum sequence length for padding
MAX_SEQUENCE_LENGTH = 50

# Load the data splits, ensuring we only get word data, time data, and labels
_, word_train, time_train, y_train = load_split(TRAIN_PATH, load_audio=False)
_, word_val, time_val, y_val = load_split(VAL_PATH, load_audio=False)
_, word_test, time_test, y_test = load_split(TEST_PATH, load_audio=False)

with open(os.path.join("pitt_split", "vocab.pkl"), "rb") as f:
    data = f.read()
vocab = pickle.loads(data)

with open(os.path.join("pitt_split", "word2vec_vectors.pkl"), "rb") as f:
    word2vec_vectors = pickle.load(f)

weight = Weights(vocab, word2vec_vectors)
embedding_vectors = weight.get_weight_matrix()
# Determine the embedding dimension dynamically from the prepared matrix
embedding_dim = embedding_vectors.shape[1]

# Create embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=embedding_dim, # FIX: Use the dynamic embedding dimension
                            weights=[embedding_vectors],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False) # Set to False to prevent re-training of pre-trained weights

# Define inputs
word_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='word_input', dtype=tf.int32)

# Define input layers
time_stamps = Input(shape=(MAX_SEQUENCE_LENGTH, 2), name='time_input', dtype=tf.float32)

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
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model.summary()

# Pad the training, validation, and test data
word_train_padded, time_train_padded = pad_sequences_and_times_np(word_train, time_train, MAX_SEQUENCE_LENGTH)
word_val_padded, time_val_padded = pad_sequences_and_times_np(word_val, time_val, MAX_SEQUENCE_LENGTH)
word_test_padded, time_test_padded = pad_sequences_and_times_np(word_test, time_test, MAX_SEQUENCE_LENGTH)

# Train with EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit([word_train_padded, time_train_padded], y_train,
          validation_data=([word_val_padded, time_val_padded], y_val),
          epochs=50,
          batch_size=16,
          shuffle=True,
          callbacks=[callback])

# Evaluate on test data
model.evaluate([word_test_padded, time_test_padded], y_test)

# Save model locally
os.makedirs("models", exist_ok=True)
# Save the model to the folder
model.save(os.path.join("models", "word_time_model.keras"))