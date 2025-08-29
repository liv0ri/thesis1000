import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling1D, LSTM, Embedding
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from utils import pad_sequences_and_times_np
import pickle
from config import VOCAB_PATH, MAX_SEQUENCE_LENGTH

if __name__ == "__main__":
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError("Vocab file not found. Please ensure it is in the project directory.")
    
    with open(VOCAB_PATH, "rb") as f:
        data = f.read()
    vocab = pickle.loads(data)
    
    audio_raw = np.random.randn(1, 16000).astype(np.float32)
    word_ids_raw = [np.random.randint(1, len(vocab) + 1, size=(5,)).tolist()] # Dummy word sequence
    time_raw = np.random.randn(1, 5, 2).astype(np.float32) # Dummy time data
    
    # Pre-process the word data for models that need it
    word_ids_padded, _ = pad_sequences_and_times_np(word_ids_raw, None, MAX_SEQUENCE_LENGTH)
    
    # Instantiate custom objects required for loading models
    custom_objects = {
        "Wav2VecFeatureExtractor": Wav2VecFeatureExtractor,
        "LSTM": LSTM,
        "Embedding": Embedding,
        "GlobalAveragePooling1D": GlobalAveragePooling1D
    }

    # ======================
    # AUDIO WORD MODEL PREDICTION
    # ======================
    print("\n--- Audio-Word Model Prediction Start ---")
    # Define the preprocessing layers. These must be re-instantiated here
    # to process the new audio data before it goes into the loaded model.
    model_checkpoint = "facebook/wav2vec2-base"
    feature_extractor = Wav2VecFeatureExtractor(model_checkpoint)
    pooling_layer = GlobalAveragePooling1D()

    # Perform the audio feature extraction and pooling first
    audio_features = feature_extractor(audio_raw)
    audio_features_pooled = pooling_layer(audio_features).numpy()

    # Load the saved model.
    loaded_audio_word_model = load_model(os.path.join("models", "best_audio_word_model_overall.keras"))
    
    # Make a prediction by passing the inputs as a list.
    prediction = loaded_audio_word_model.predict([audio_features_pooled, word_ids_padded])

    # The output is a probability. Convert it to a binary class.
    predicted_class = (prediction > 0.5).astype(int)

    print("\n--- Audio-Word Model Prediction Results ---")
    print(f"Prediction probability: {prediction[0][0]:.4f}")
    print(f"Predicted class: {predicted_class[0][0]}")

    # ======================
    # AUDIO MODEL PREDICTION
    # ======================
    print("\n--- Audio-Only Model Prediction Start ---")
    loaded_audio_model = load_model(os.path.join("models", "best_model_audio_overall.keras"))
    prediction = loaded_audio_model.predict(audio_features_pooled)
    predicted_class = (prediction > 0.5).astype(int)
    print("\n--- Audio-Only Model Prediction Results ---")
    print(f"Prediction probability: {prediction[0][0]:.4f}")
    print(f"Predicted class: {predicted_class[0][0]}")

    # ======================
    # WORD TIME MODEL PREDICTION
    # ======================
    print("\n--- Word-Time Model Prediction Start ---")
    word_time_new = np.random.randint(1, len(vocab) + 1, size=(1, MAX_SEQUENCE_LENGTH))
    time_new = np.random.randn(1, MAX_SEQUENCE_LENGTH, 2).astype(np.float32)
    loaded_word_time_model = load_model(os.path.join("models", "best_text_time_model_overall.keras"))
    prediction = loaded_word_time_model.predict([word_time_new, time_new])
    predicted_class = (prediction > 0.5).astype(int)
    print("\n--- Word-Time Model Prediction Results ---")
    print(f"Prediction probability: {prediction[0][0]:.4f}")
    print(f"Predicted class: {predicted_class[0][0]}")

    # ======================
    # TEXT MODEL PREDICTION
    # ======================
    loaded_text_model = load_model(os.path.join("models", "best_text_model_overall.keras"))
    prediction_prob = loaded_text_model.predict(word_ids_padded)
    predicted_class = (prediction_prob > 0.5).astype(int)
    print("\n--- Text-Only Model Prediction Results ---")
    print(f"Prediction probability: {prediction_prob[0][0]:.4f}")
    print(f"Predicted class: {predicted_class[0][0]}")

    # ======================
    # AUDIO TIME MODEL PREDICTION
    # ======================
    loaded_audio_time_model = load_model(os.path.join("models", "best_audio_time_model_overall.keras"))
    prediction = loaded_audio_time_model.predict([audio_features_pooled, time_new])
    predicted_class = (prediction > 0.5).astype(int)
    print("\n--- Audio-Time Model Prediction Results ---")
    print(f"Prediction probability: {prediction[0][0]:.4f}")
    print(f"Predicted class: {predicted_class[0][0]}")
    
    # ======================
    # COMBINED MODEL PREDICTION
    # ======================
    loaded_combined_model = load_model(os.path.join("models", "best_model_audio_text_time.keras"))
    prediction = loaded_combined_model.predict([audio_features_pooled, word_ids_padded, time_new])
    predicted_class = (prediction > 0.5).astype(int)
    print("\n--- Combined Model Prediction Results ---")
    print(f"Prediction probability: {prediction[0][0]:.4f}")
    print(f"Predicted class: {predicted_class[0][0]}")