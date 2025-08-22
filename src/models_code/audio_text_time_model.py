import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from weights import Weights
from utils import load_split
import pickle
from config import TRAIN_PATH, TEST_PATH, VAL_PATH


audio_train, word_train, time_train, y_train = load_split(TRAIN_PATH)
audio_val, word_val, time_val, y_val = load_split(VAL_PATH)
audio_test, word_test, time_test, y_test = load_split(TEST_PATH)

# Load the pre-trained model using the Hugging Face interface
model_checkpoint = "facebook/wav2vec2-base"
# Define the inputs to the model
audio_input = Input(shape=(16000,), dtype=tf.float32)

audio_features = Wav2VecFeatureExtractor(model_checkpoint)(audio_input)
audio_output = GlobalAveragePooling1D()(audio_features)
audio_output = Dropout(0.5)(audio_output)

# Create the TensorFlow functional API model
audio_model = Model(inputs=audio_input, outputs=audio_output, name="audio_model")

# Print the model summary
audio_model.summary()

with open(os.path.join("pitt_split", "vocab.pkl"), "rb") as f:
    data = f.read()
vocab = pickle.loads(data)

with open(os.path.join("pitt_split", "word2vec_vectors.pkl"), "rb") as f:
    word2vec_vectors = pickle.load(f)

weight = Weights(vocab, word2vec_vectors)

embedding_vectors = weight.get_weight_matrix()

embedding_layer = Embedding(input_dim=len(vocab) + 1,
                             output_dim=300,
                             weights=[embedding_vectors],
                             input_length=50,
                             trainable=False)

word_input = Input(shape=(50,), dtype=tf.int32)
time_stamps = Input(shape=(50, 2), dtype=tf.float32)

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])
# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)
# Define the model
word_time_model = Model(inputs=[word_input, time_stamps], outputs=lstm_output, name="word_model")

# Print the model summary
word_time_model.summary()

combined_output = Concatenate()([audio_model.output, word_time_model.output])

final_output = Dense(1, activation='sigmoid')(combined_output)
combined_model = Model(inputs=[audio_model.input, word_time_model.input], outputs=final_output, name="combined_model")
combined_model.summary()

combined_train = [word_train, time_train]
combined_val = [word_val, time_val]
combined_test = [word_test, time_test]

# Compile and train the model
combined_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
combined_model.fit([audio_train, combined_train], y_train,
                   epochs=50, batch_size=16, validation_data=([audio_val, combined_val], y_val),
                  callbacks=[callback])

# Evaluate the model using the test set
combined_model.evaluate([audio_test, combined_test], y_test)

os.makedirs("models", exist_ok=True)

# Save the model to the models directory
combined_model.save(os.path.join("models", "audio_word_time_model.keras"))
