import numpy as np
import tensorflow as tf
import os
from keras.layers import Dense, LSTM,  Dropout, Embedding, Input, GlobalAveragePooling1D, Concatenate
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from tensorflow.keras.models import Model
from utils import load_split

audio_train, _, time_train, y_train = load_split("pitt_split/train", load_words=False)
audio_val, _, time_val, y_val = load_split("pitt_split/val", load_words=False)
audio_test, _, time_test, y_test = load_split("pitt_split/test", load_words=False)

model_checkpoint = "facebook/wav2vec2-base"
input_values = Input(shape=(16000,), dtype=tf.float32)

# huggingface_model = TFAutoModel.from_pretrained(model_checkpoint, trainable=False, from_pt=True)
# Pass the inputs through the Wav2Vec model
# wav2vec_output = huggingface_model(input_values)
# BEFORE WE HAD THIS INSTEAD
# Define the inputs to the model
audio_features = Wav2VecFeatureExtractor(model_checkpoint)(input_values)
audio_output = GlobalAveragePooling1D()(audio_features)
audio_output = Dropout(0.5)(audio_output)

audio_model = Model(inputs=input_values, outputs=audio_output, name='audio_model')
# Print the model summary
audio_model.summary()

# Time inputs
time_stamps = Input(shape=(50, 2), name='time_input')

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)

# Time model
time_model = Model(inputs=time_stamps, outputs=lstm_output, name='time_model')

# Print the model summary
time_model.summary()

# IN CASE WWE REMOVE THE ONES ABOVE
# Reshape word_model output to match the shape of audio_model output
# audio_model_output = layers.GlobalAveragePooling1D()(audio_model.output[1])
# # Drop-out layer before the final Classification-Head
# audio_model_output = layers.Dropout(0.5) (audio_model_output)


concatenated_output = Concatenate()([audio_model.output, time_model.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
combined = Dense(1, activation='sigmoid')(concatenated_output)
audio_time_model = Model(inputs=[audio_model.input, time_model.input], outputs=combined, name='audio_time_model')
audio_time_model.summary()

# Compile and train the model
audio_time_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)

# Train
audio_time_model.fit([audio_train, time_train], y_train,
                   epochs=50, batch_size=16,
validation_data=([audio_val, time_val], y_val),
                   callbacks=[callback])

# Evaluate
audio_time_model.evaluate([audio_val, time_val], y_val)

# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model to the folder
audio_time_model.save(os.path.join("models", "audio_time_model.keras"))
