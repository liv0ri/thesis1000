import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from utils import load_split, pad_sequences_and_times_np
from config import TRAIN_PATH, TEST_PATH, VAL_PATH
import pickle
from weights import Weights

MAX_SEQUENCE_LENGTH = 50

audio_train, word_train, time_train, y_train = load_split(TRAIN_PATH)
audio_val, word_val, time_val, y_val = load_split(VAL_PATH)
audio_test, word_test, time_test, y_test = load_split(TEST_PATH)

# Load the pre-trained Wav2Vec model using the Hugging Face interface
model_checkpoint = "facebook/wav2vec2-base"

# Define the inputs to the model
audio_input = Input(shape=(16000,), dtype=tf.float32)

# Use the Wav2Vec feature extractor to get audio features
audio_features = Wav2VecFeatureExtractor(model_checkpoint)(audio_input)
# Apply global average pooling to the audio features
audio_output = GlobalAveragePooling1D()(audio_features)
# Add a dropout layer for regularization
audio_output = Dropout(0.5)(audio_output)

# Create the TensorFlow functional API model for audio
audio_model = Model(inputs=audio_input, outputs=audio_output, name="audio_model")

audio_model.summary()

# Load the pre-processed vocabulary and word2vec vectors
with open(os.path.join("pitt_split", "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

with open(os.path.join("pitt_split", "word2vec_vectors.pkl"), "rb") as f:
    word2vec_vectors = pickle.load(f)

weight = Weights(vocab, word2vec_vectors)

embedding_vectors = weight.get_weight_matrix()

# Determine the embedding dimension dynamically from the prepared matrix
embedding_dim = embedding_vectors.shape[1]

# Create the embedding layer with the pre-trained weights
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=embedding_dim,
                            weights=[embedding_vectors],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False) # Set to False to prevent re-training of pre-trained weights

# Define the inputs for the word and time model
word_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
time_stamps = Input(shape=(MAX_SEQUENCE_LENGTH, 2), dtype=tf.float32)

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])
# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)
# Define the model
word_time_model = Model(inputs=[word_input, time_stamps], outputs=lstm_output, name="word_model")

word_time_model.summary()

# Concatenate the outputs of the audio and word/time models
combined_output = Concatenate()([audio_model.output, word_time_model.output])

# Add a final dense layer for classification
final_output = Dense(1, activation='sigmoid')(combined_output)

# Build the combined model
combined_model = Model(
    inputs=[audio_model.input, *word_time_model.input],  # unpack list
    outputs=final_output,
    name="combined_model"
)

combined_model.summary()

# Pad the training, validation, and test data
word_train_padded, time_train_padded = pad_sequences_and_times_np(word_train, time_train, MAX_SEQUENCE_LENGTH)
word_val_padded, time_val_padded = pad_sequences_and_times_np(word_val, time_val, MAX_SEQUENCE_LENGTH)
word_test_padded, time_test_padded = pad_sequences_and_times_np(word_test, time_test, MAX_SEQUENCE_LENGTH)

# Prepare data for training/validation/testing (flattened, not nested)
combined_train_inputs = [audio_train, word_train_padded, time_train_padded]
combined_val_inputs = [audio_val, word_val_padded, time_val_padded]
combined_test_inputs = [audio_test, word_test_padded, time_test_padded]
combined_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
                                            
# Train the model
combined_model.fit(combined_train_inputs, y_train,
                   epochs=50, batch_size=16, 
                   validation_data=(combined_val_inputs, y_val),
                   callbacks=[callback])

combined_model.evaluate(combined_test_inputs, y_test)

# Create the directory to save the model if it doesn't exist
os.makedirs("models", exist_ok=True)

combined_model.save(os.path.join("models", "audio_word_time_model.keras"))
