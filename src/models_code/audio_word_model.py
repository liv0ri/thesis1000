import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from wav2vec_feature_extractor import Wav2VecFeatureExtractor  # your custom wrapper
import os
from weights import Weights

vocab = {f"word{i}": i for i in range(1, 101)}
word2vec_vectors = {word: np.random.rand(300) for word in vocab}

# Dummy data (10 train, 2 val, 2 test)
audio_train = np.random.randn(10, 16000).astype(np.float32)
word_train = np.random.randint(1, len(vocab) + 1, size=(10, 50))
y_train = np.random.randint(0, 2, size=(10, 1))

audio_val = np.random.randn(2, 16000).astype(np.float32)
word_val = np.random.randint(1, len(vocab) + 1, size=(2, 50))
y_val = np.random.randint(0, 2, size=(2, 1))

audio_test = np.random.randn(2, 16000).astype(np.float32)
word_test = np.random.randint(1, len(vocab) + 1, size=(2, 50))
y_test = np.random.randint(0, 2, size=(2, 1))

# Prepare embedding matrix from your weights class
weight = Weights(vocab, word2vec_vectors)
embedding_vectors = weight.get_weight_matrix()

# Create the embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=300,
                            weights=[embedding_vectors],
                            input_length=50,
                            trainable=False)

# AUDIO MODEL BRANCH
model_checkpoint = "facebook/wav2vec2-base"
audio_input = Input(shape=(16000,), name="audio_input")

# Extract features with your wav2vec wrapper (assumed returns tensor with shape (batch, seq_len, feat_dim))
wav2vec_extractor = Wav2VecFeatureExtractor(model_checkpoint)
audio_features = wav2vec_extractor(audio_input)

# Pool audio features to fixed length vector
audio_pooled = GlobalAveragePooling1D()(audio_features)
audio_pooled = Dropout(0.5)(audio_pooled)

# WORD MODEL BRANCH
word_input = Input(shape=(50,), dtype=tf.int32, name="word_input")
word_embedded = embedding_layer(word_input)

# LSTM over word embeddings
word_lstm = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(word_embedded)

# CONCATENATE audio and word outputs
combined = Concatenate()([audio_pooled, word_lstm])

# Final dense layer for binary classification
output = Dense(1, activation='sigmoid')(combined)

# Define full model with two inputs
model = Model(inputs=[audio_input, word_input], outputs=output, name="audio_word_model")

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# Print summary to verify structure
model.summary()

# Train with EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit([audio_train, word_train], y_train,
          validation_data=([audio_val, word_val], y_val),
          epochs=50,
          batch_size=16,
          callbacks=[callback],
          shuffle=True)

# Evaluate on test data
model.evaluate([audio_test, word_test], y_test)

# Save model
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "audio_word_model.keras"))
