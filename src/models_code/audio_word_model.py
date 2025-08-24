import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D, Concatenate, Embedding
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from tensorflow.keras.models import Model
from utils import load_split
from config import TRAIN_PATH, TEST_PATH, VAL_PATH
from weights import Weights
import pickle

# Define the maximum sequence length for padding
MAX_SEQUENCE_LENGTH = 50

# Load the data splits, ensuring we only get audio and word data
audio_train, word_train, _, y_train = load_split(TRAIN_PATH, load_times=False)
audio_val, word_val, _, y_val = load_split(VAL_PATH, load_times=False)
audio_test, word_test, _, y_test = load_split(TEST_PATH, load_times=False)

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

# Create the embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=embedding_dim, # FIX: Use the dynamic embedding dimension
                            weights=[embedding_vectors],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

model_checkpoint = "facebook/wav2vec2-base"
audio_input = Input(shape=(16000,), name="audio_input")

# Extract features with your wav2vec wrapper 
wav2vec_extractor = Wav2VecFeatureExtractor(model_checkpoint)
audio_features = wav2vec_extractor(audio_input)

# Pool audio features to fixed length vector
audio_pooled = GlobalAveragePooling1D()(audio_features)
audio_pooled = Dropout(0.5)(audio_pooled)

# WORD MODEL BRANCH
word_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="word_input")
word_embedded = embedding_layer(word_input)

# LSTM over word embeddings
word_lstm = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(word_embedded)

# CONCATENATE audio and word outputs
combined = Concatenate()([audio_pooled, word_lstm])

# Final dense layer for binary classification
output = Dense(1, activation='sigmoid')(combined)

# Define full model with two inputs
model = Model(inputs=[audio_input, word_input], outputs=output, name="audio_word_model")

# Print summary to verify structure
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

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# Print summary to verify structure
model.summary()

# Train with EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with the padded word data
model.fit([audio_train, word_train_padded], y_train,
          validation_data=([audio_val, word_val_padded], y_val),
          epochs=50,
          batch_size=16,
          callbacks=[callback],
          shuffle=True)

# Evaluate on test data
model.evaluate([audio_test, word_test_padded], y_test)

# Save model
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "audio_word_model.keras"))
