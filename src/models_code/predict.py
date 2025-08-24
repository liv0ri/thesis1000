
import os
import numpy as np
import tensorflow as tf
from wav2vec_feature_extractor import Wav2VecFeatureExtractor 
import pickle

with open(os.path.join("pitt_split", "vocab.pkl"), "rb") as f:
    data = f.read()
vocab = pickle.loads(data)


Word2VecFeatureExtractor = Wav2VecFeatureExtractor("facebook/wav2vec2-base")
# ======================
# AUDIO MODEL
# ======================
# Assuming you have a new audio file to predict on
# dummy audio_new of shape (1, 16000)
print("Loading audio_model...")
audio_new = np.random.randn(1, 16000).astype(np.float32)

# Load the saved model
loaded_audio_model = tf.keras.models.load_model(os.path.join("models", "audio_model.keras"))

# Make a prediction
prediction = loaded_audio_model.predict(audio_new)

# The output is a probability. You can convert it to a binary class.
predicted_class = (prediction > 0.5).astype(int)

print(f"Prediction probability: {prediction[0][0]:.4f}")
print(f"Predicted class: {predicted_class[0][0]}")


# ======================
# WORD TIME MODEL
# ======================
# Assuming new data for a single sample
# dummy word_new with shape (1, 50) and time_new with shape (1, 50, 2)
print("Loading word_time_model...")
word_new = np.random.randint(1, 1000 + 1, size=(1, 50))
word_new = np.clip(word_new, 0, 101 - 1)
time_new = np.random.randn(1, 50, 2).astype(np.float32)

# Load the saved model
loaded_word_time_model = tf.keras.models.load_model(os.path.join("models", "word_time_model.keras"))

# The inputs must be a list
prediction = loaded_word_time_model.predict([word_new, time_new])

predicted_class = (prediction > 0.5).astype(int)

print(f"Prediction probability: {prediction[0][0]:.4f}")
print(f"Predicted class: {predicted_class[0][0]}")


# ======================
# TEXT MODEL PREDICTION
# ======================
# Assuming you have new word sequence data to predict on
# dummy word_new with shape (1, 50)
print("Loading text_model...")
word_new = np.random.randint(1, len(vocab) + 1, size=(1, 50))
word_new = np.clip(word_new, 0, 101 - 1)

# Load the saved model
loaded_text_model = tf.keras.models.load_model(os.path.join("models", "text_model.keras"))

# Make a prediction
prediction_prob = loaded_text_model.predict(word_new)

# The output is a probability. You can convert it to a binary class.
predicted_class = (prediction_prob > 0.5).astype(int)

print(f"Prediction probability: {prediction_prob[0][0]:.4f}")
print(f"Predicted class: {predicted_class[0][0]}")

# ======================
# AUDIO TIME MODEL
# ======================
# Assuming new data for a single sample
# dummy audio_new with shape (1, 16000) and time_new with shape (1, 50, 2)
print("Loading audio_time_model...")
audio_new = np.random.randn(1, 16000).astype(np.float32)
time_new = np.random.randn(1, 50, 2).astype(np.float32)

# Load the saved model
loaded_audio_time_model = tf.keras.models.load_model(os.path.join("models", "audio_time_model.keras"))

# The inputs must be a list
prediction = loaded_audio_time_model.predict([audio_new, time_new])

predicted_class = (prediction > 0.5).astype(int)

print(f"Prediction probability: {prediction[0][0]:.4f}")
print(f"Predicted class: {predicted_class[0][0]}")


# ======================
# AUDIO WORD MODEL
# ======================
print("\n--- Prediction Start ---")
print("Loading audio_word_model...")
audio_new = np.random.randn(1, 16000).astype(np.float32)
word_new = np.random.randint(1, len(vocab) + 1, size=(1, 50))


# IMPORTANT: The custom Wav2VecFeatureExtractor must be in scope
# when loading the model. We pass it via the custom_objects dictionary.
loaded_audio_word_model = tf.keras.models.load_model(os.path.join("models", "audio_word_model.keras")    )

# Make a prediction by passing the inputs as a list
prediction = loaded_audio_word_model.predict([audio_new, word_new])
# The output is a probability. Convert it to a binary class.
predicted_class = (prediction > 0.5).astype(int)

print("\n--- Prediction Results ---")
print(f"Prediction probability: {prediction[0][0]:.4f}")
print(f"Predicted class: {predicted_class[0][0]}")


# ======================
# COMBINED MODEL
# ======================
# Assuming new data for a single sample
# dummy audio_new (1, 16000), word_new (1, 50), time_new (1, 50, 2)
print("Loading audio_word_time_model...")
# Combined model prediction fix
audio_new = np.random.randn(1, 16000).astype(np.float32)
word_new = np.random.randint(1, len(vocab), size=(1, 50)).astype(np.int32) 
word_new = np.clip(word_new, 0, 101 - 1)
time_new = np.random.randn(1, 50, 2).astype(np.float32)

loaded_combined_model = tf.keras.models.load_model(os.path.join("models", "audio_word_time_model.keras"))

# Pass all inputs as a flat list
prediction = loaded_combined_model.predict([audio_new, word_new, time_new])

predicted_class = (prediction > 0.5).astype(int)

print(f"Prediction probability: {prediction[0][0]:.4f}")
print(f"Predicted class: {predicted_class[0][0]}")