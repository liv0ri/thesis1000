import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D, Concatenate
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from tensorflow.keras.models import Model
from utils import load_split
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
# Print the model summary
audio_model.summary()

time_stamps = Input(shape=(MAX_SEQUENCE_LENGTH, 2), name='time_input', dtype=tf.float32)

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)

# Time model
time_model = Model(inputs=time_stamps, outputs=lstm_output, name='time_model')

# Print the model summary
time_model.summary()

concatenated_output = Concatenate()([audio_model.output, time_model.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
combined = Dense(1, activation='sigmoid')(concatenated_output)
audio_time_model = Model(inputs=[audio_model.input, time_model.input], outputs=combined, name='audio_time_model')
audio_time_model.summary()

# FIX: Pad the time sequences to a uniform length before passing them to the model
def pad_time_sequences(time_sequences, maxlen):
    """
    Pads time sequences to a uniform length and converts them to a numpy array.
    """
    padded_times = []
    for times in time_sequences:
        # FIX: Ensure each item is a valid sequence of length 2 before cleaning and padding
        cleaned_times = []
        for item in times:
            # Check if the item is not None and has a length of 2 (e.g., a list or tuple of two elements)
            if item is not None and isinstance(item, (list, tuple)) and len(item) == 2:
                cleaned_times.append(item)
        
        # Determine the length of the valid sequence (up to maxlen)
        seq_len = min(len(cleaned_times), maxlen)
        
        # Create a padded sequence
        padded_time_seq = np.zeros((maxlen, 2), dtype='float32')
        if seq_len > 0:
            # Convert the cleaned list to a NumPy array before assignment
            cleaned_times_np = np.array(cleaned_times, dtype='float32')
            padded_time_seq[:seq_len, :] = cleaned_times_np[:seq_len, :]
        
        padded_times.append(padded_time_seq)
            
    return np.array(padded_times)

time_train_padded = pad_time_sequences(time_train, MAX_SEQUENCE_LENGTH)
time_val_padded = pad_time_sequences(time_val, MAX_SEQUENCE_LENGTH)
time_test_padded = pad_time_sequences(time_test, MAX_SEQUENCE_LENGTH)

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

# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model to the folder
audio_time_model.save(os.path.join("models", "audio_time_model.keras"))
