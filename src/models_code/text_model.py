import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from weights import Weights
from utils import load_split
import pickle
from config import TRAIN_PATH, TEST_PATH, VAL_PATH

# Define the maximum sequence length for padding
MAX_SEQUENCE_LENGTH = 50

# Load the data splits, ensuring we only get word data and labels
_, word_train, _, y_train = load_split(TRAIN_PATH, load_audio=False, load_times=False)
_, word_val, _, y_val = load_split(VAL_PATH, load_audio=False, load_times=False)
_, word_test, _, y_test = load_split(TEST_PATH, load_audio=False, load_times=False)

# Load the pre-processed vocabulary and word2vec vectors
with open(os.path.join("pitt_split", "vocab.pkl"), "rb") as f:
    data = f.read()
vocab = pickle.loads(data)

with open(os.path.join("pitt_split", "word2vec_vectors.pkl"), "rb") as f:
    word2vec_vectors = pickle.load(f)

# Prepare embedding matrix from your weights class
weight = Weights(vocab, word2vec_vectors)
embedding_vectors = weight.get_weight_matrix()

# Determine the embedding dimension dynamically from the prepared matrix
embedding_dim = embedding_vectors.shape[1]

# Create embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=embedding_dim,  # Use the dynamic embedding dimension
                            weights=[embedding_vectors],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False) # Set to False to prevent re-training of pre-trained weights

# Define input layer
word_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='word_input', dtype=tf.int32)

# Embed input
word_embedded = embedding_layer(word_input)

# LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(word_embedded)

# Apply dense layer for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

# Build model
model = Model(inputs=word_input, outputs=output, name='word_lstm_model')

# Print the model summary
model.summary()

# FIX: Pad the word sequences to a uniform length before passing them to the model
def pad_word_sequences(word_sequences, maxlen):
    """
    Pads word sequences to a uniform length and converts them to a numpy array.
    This ensures that all input sequences have the same length as required by the
    Embedding and LSTM layers.
    """
    padded_words = []
    for words in word_sequences:
        # Ensure only integer items are kept and pad the rest with zeros
        cleaned_words = [item for item in words if isinstance(item, int)]
        seq_len = min(len(cleaned_words), maxlen)
        padded_seq = np.zeros((maxlen,), dtype='int32')
        if seq_len > 0:
            padded_seq[:seq_len] = cleaned_words[:seq_len]
        padded_words.append(padded_seq)
    return np.array(padded_words)

# Pad the training, validation, and test data
word_train_padded = pad_word_sequences(word_train, MAX_SEQUENCE_LENGTH)
word_val_padded = pad_word_sequences(word_val, MAX_SEQUENCE_LENGTH)
word_test_padded = pad_word_sequences(word_test, MAX_SEQUENCE_LENGTH)

# Compile the model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# Early stopping callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(word_train_padded, y_train,
          validation_data=(word_val_padded, y_val),
          epochs=10,
          batch_size=4,
          shuffle=True,
          callbacks=[callback])

# Evaluate on test set
print("Evaluating the model on the test set...")
model.evaluate(word_test_padded, y_test)

# Save locally in 'models' folder
os.makedirs("models", exist_ok=True)
# Save the model to the folder
model.save(os.path.join("models", "text_model.keras"))
