# DONE

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import numpy as np
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
import os

# Dummy audio and labels
audio_train = np.random.randn(10, 16000).astype(np.float32)
y_train = np.random.randint(0, 2, size=(10, 1))

audio_val = np.random.randn(2, 16000).astype(np.float32)
y_val = np.random.randint(0, 2, size=(2, 1))

audio_test = np.random.randn(2, 16000).astype(np.float32)
y_test = np.random.randint(0, 2, size=(2, 1))# Dummy audio and labels
audio_train = np.random.randn(10, 16000).astype(np.float32)
y_train = np.random.randint(0, 2, size=(10, 1))

audio_val = np.random.randn(2, 16000).astype(np.float32)
y_val = np.random.randint(0, 2, size=(2, 1))

audio_test = np.random.randn(2, 16000).astype(np.float32)
y_test = np.random.randint(0, 2, size=(2, 1))

# Load the pre-trained model using the Hugging Face interface
model_checkpoint = "facebook/wav2vec2-base"

# OLD CODE THAT WAS CHANGED
# huggingface_model = TFAutoModel.from_pretrained(model_checkpoint, trainable=False, from_pt=True)

# # Define the inputs to the model
# input_values = tf.keras.Input(shape=(16000,), dtype=tf.float32)

# # Pass the inputs through the Wav2Vec model
# wav2vec_output = huggingface_model(input_values)

# # Reshape word_model output to match the shape of audio_model output
# audio_model_output = layers.GlobalAveragePooling1D()(wav2vec_output[1])

input_values = tf.keras.Input(shape=(16000,), dtype=tf.float32)
wav2vec_features = Wav2VecFeatureExtractor(model_checkpoint)(input_values)

# Reshape word_model output to match the shape of audio_model output
audio_model_output = layers.GlobalAveragePooling1D()(wav2vec_features)
# Drop-out layer before the final Classification-Head
audio_model_output = layers.Dropout(0.5) (audio_model_output)

output = Dense(1, activation='sigmoid')(audio_model_output)

# Create the TensorFlow functional API model
audio_model = tf.keras.Model(inputs=input_values, outputs=output)

audio_model.summary()

# Train the audio model
audio_model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
audio_model.fit(audio_train, y_train,
                       epochs=50, batch_size=16, validation_data=(audio_val, y_val),
                       callbacks=callback)

# Evaluate the audio model
audio_model.evaluate(audio_test, y_test)

# Save the model to the models directory
audio_model.save(os.path.join("models", "audio_model.keras"))