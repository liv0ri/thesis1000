import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling1D, LSTM, Embedding
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from utils import pad_sequences_and_times_np
from config import VOCAB_PATH, MAX_SEQUENCE_LENGTH


class DementiaPredictor:
    def __init__(self, model_dir="models/both"):
        # Load vocab
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError("Vocab file not found. Please ensure it is in the project directory.")
        with open(VOCAB_PATH, "rb") as f:
            self.vocab = pickle.loads(f.read())

        # Audio feature extractor
        self.feature_extractor = Wav2VecFeatureExtractor("facebook/wav2vec2-base")
        self.pooling_layer = GlobalAveragePooling1D()

        # Load models
        self.models = {
            "audio": load_model(os.path.join(model_dir, "best_audio_model.keras")),
            "text": load_model(os.path.join(model_dir, "best_text_model.keras")),
            "audio_text": load_model(os.path.join(model_dir, "best_audio_text_model.keras")),
            "audio_time": load_model(os.path.join(model_dir, "best_audio_time_model.keras")),
            "text_time": load_model(os.path.join(model_dir, "best_text_time_model.keras")),
            "audio_text_time": load_model(os.path.join(model_dir, "best_audio_text_time_model.keras")),
        }

    def preprocess_audio(self, audio_raw):
        features = self.feature_extractor(audio_raw)
        return self.pooling_layer(features).numpy()

    def preprocess_text(self, word_ids_raw):
        word_ids_padded, _ = pad_sequences_and_times_np(word_ids_raw, None, MAX_SEQUENCE_LENGTH)
        return word_ids_padded

    def predict_audio(self, audio_raw):
        audio_features = self.preprocess_audio(audio_raw)
        pred = self.models["audio"].predict(audio_features)
        return float(pred[0][0]), int(pred[0][0] > 0.5)

    def predict_text(self, word_ids_raw):
        word_ids_padded = self.preprocess_text(word_ids_raw)
        pred = self.models["text"].predict(word_ids_padded)
        return float(pred[0][0]), int(pred[0][0] > 0.5)

    def predict_audio_text(self, audio_raw, word_ids_raw):
        audio_features = self.preprocess_audio(audio_raw)
        word_ids_padded = self.preprocess_text(word_ids_raw)
        pred = self.models["audio_text"].predict([audio_features, word_ids_padded])
        return float(pred[0][0]), int(pred[0][0] > 0.5)

    def predict_text_time(self, word_ids_raw, time_data):
        word_ids_padded = self.preprocess_text(word_ids_raw)
        pred = self.models["text_time"].predict([word_ids_padded, time_data])
        return float(pred[0][0]), int(pred[0][0] > 0.5)

    def predict_audio_time(self, audio_raw, time_data):
        audio_features = self.preprocess_audio(audio_raw)
        pred = self.models["audio_time"].predict([audio_features, time_data])
        return float(pred[0][0]), int(pred[0][0] > 0.5)

    def predict_combined(self, audio_raw, word_ids_raw, time_data):
        audio_features = self.preprocess_audio(audio_raw)
        word_ids_padded = self.preprocess_text(word_ids_raw)
        pred = self.models["audio_text_time"].predict([audio_features, word_ids_padded, time_data])
        return float(pred[0][0]), int(pred[0][0] > 0.5)


if __name__ == "__main__":
    predictor = DementiaPredictor()

    # Dummy data
    audio_raw = np.random.randn(1, 16000).astype(np.float32)
    word_ids_raw = [np.random.randint(1, len(predictor.vocab) + 1, size=(5,)).tolist()]
    time_raw = np.random.randn(1, MAX_SEQUENCE_LENGTH, 2).astype(np.float32)

    print("Audio only:", predictor.predict_audio(audio_raw))
    print("Text only:", predictor.predict_text(word_ids_raw))
    print("Audio + Text:", predictor.predict_audio_text(audio_raw, word_ids_raw))
    print("Text + Time:", predictor.predict_text_time(word_ids_raw, time_raw))
    print("Audio + Time:", predictor.predict_audio_time(audio_raw, time_raw))
    print("All combined:", predictor.predict_combined(audio_raw, word_ids_raw, time_raw))
