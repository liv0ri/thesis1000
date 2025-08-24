import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D, Concatenate
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from tensorflow.keras.models import Model
from utils import load_split, pad_sequences_and_times_np
from config import TRAIN_PATH, TEST_PATH, VAL_PATH

MAX_SEQUENCE_LENGTH = 50

audio_train, _, time_train, y_train = load_split(TRAIN_PATH, load_words=False)
audio_val, _, time_val, y_val = load_split(VAL_PATH, load_words=False)
audio_test, _, time_test, y_test = load_split(TEST_PATH, load_words=False)

model_checkpoint = "facebook/wav2vec2-base"
input_values = Input(shape=(16000,), dtype=tf.float32)

# Define the inputs to the model
audio_features = Wav2VecFeatureExtractor(model_checkpoint)(input_values)
audio_output = GlobalAveragePooling1D()(audio_features)
audio_output = Dropout(0.5)(audio_output)

audio_model = Model(inputs=input_values, outputs=audio_output, name='audio_model')
audio_model.summary()

time_stamps = Input(shape=(MAX_SEQUENCE_LENGTH, 2), name='time_input', dtype=tf.float32)

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)

# Time model
time_model = Model(inputs=time_stamps, outputs=lstm_output, name='time_model')

time_model.summary()

concatenated_output = Concatenate()([audio_model.output, time_model.output])
# combined outputs
combined = Dense(1, activation='sigmoid')(concatenated_output)
audio_time_model = Model(inputs=[audio_model.input, time_model.input], outputs=combined, name='audio_time_model')
audio_time_model.summary()

_, time_train_padded = pad_sequences_and_times_np(None, time_train, MAX_SEQUENCE_LENGTH)
_, time_val_padded = pad_sequences_and_times_np(None, time_val, MAX_SEQUENCE_LENGTH)
_, time_test_padded = pad_sequences_and_times_np(None, time_test, MAX_SEQUENCE_LENGTH)
 
# Compile and train the model
audio_time_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)

# Train the model with the padded time data
audio_time_model.fit([audio_train, time_train_padded], y_train,
                     epochs=50, batch_size=16,
                     validation_data=([audio_val, time_val_padded], y_val),
                     callbacks=[callback])

audio_time_model.evaluate([audio_test, time_test_padded], y_test)

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model to the folder
audio_time_model.save(os.path.join("models", "audio_time_model.keras"))
