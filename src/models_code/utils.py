import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling1D
import tensorflow as tf
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from config import FEATURE_EXTRACTION_BATCH_SIZE

def pad_sequences_and_times_np(word_sequences=None, time_sequences=None, maxlen=100):
    """Pads word and time sequences to a fixed length."""
    num_samples = len(word_sequences) if word_sequences is not None else len(time_sequences)
    padded_words_np = np.zeros((num_samples, maxlen), dtype="int32")
    if word_sequences is not None:
        for i, seq in enumerate(word_sequences):
            seq_len = min(len(seq), maxlen)
            if seq_len > 0:
                padded_words_np[i, :seq_len] = seq[:seq_len]

    padded_times_np = np.zeros((num_samples, maxlen, 2), dtype="float32")
    if time_sequences is not None:
        for i, seq in enumerate(time_sequences):
            seq_len = min(len(seq), maxlen)
            if seq_len > 0:
                seq_array = np.array(seq[:seq_len], dtype="float32")
                if seq_array.shape[1] != 2:
                    raise ValueError(f"Expected shape (_,2), got {seq_array.shape}")
                padded_times_np[i, :seq_len, :] = seq_array
    
    return padded_words_np, padded_times_np

def prepare_audio_data(data_points, features_cache_path, labels_cache_path):
    raw_audios = np.array([d['audio'] for d in data_points])
    all_labels = np.array([1 if d['label'] == 'dementia' else 0 for d in data_points])
    
    # Instantiate the feature extractor
    feature_extractor = Wav2VecFeatureExtractor("facebook/wav2vec2-base")
    pooling_layer = GlobalAveragePooling1D()
    
    print(f"Processing audio files in batches of {FEATURE_EXTRACTION_BATCH_SIZE}...")
    extracted_features_list = []
    for i in range(0, len(raw_audios), FEATURE_EXTRACTION_BATCH_SIZE):
        batch_raw_audios = raw_audios[i:i + FEATURE_EXTRACTION_BATCH_SIZE]
        
        # Process and extract features for the current batch
        batch_features = feature_extractor(batch_raw_audios)
        
        # Apply GlobalAveragePooling1D to get fixed-size vectors
        batch_features_pooled = pooling_layer(batch_features)
        
        extracted_features_list.append(batch_features_pooled)
        print(f"Processed batch {i // FEATURE_EXTRACTION_BATCH_SIZE + 1} of {len(raw_audios) // FEATURE_EXTRACTION_BATCH_SIZE + 1}...")

    # Concatenate all the batches into a single tensor
    all_audios = tf.concat(extracted_features_list, axis=0)
    
    all_audios = all_audios.numpy()
    
    # Cache the extracted features to disk
    np.save(features_cache_path, all_audios)
    np.save(labels_cache_path, all_labels)
    return all_audios, all_labels