import numpy as np
import soundfile as sf
import os
import pickle
import warnings

def load_audio_file(file_path, target_length=16000):
    audio, sr = sf.read(file_path)
    
    # Handle multi-channel audio
    if audio.ndim > 1:
        warnings.warn(f"Audio file {file_path} has multiple channels. Using the first channel.")
        audio = audio[:, 0]
        
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))
        
    return audio.astype(np.float32)


def load_split(root_dir, target_length=16000, load_audio=True, load_words=True, load_times=True):
    split_dir = root_dir
    audio_data, labels, all_word_lists, all_time_lists = [], [], [], []

    label_map = {"control": 0, "dementia": 1}

    # --- First Pass: Find max lengths for padding ---
    max_word_outer = 0
    max_word_inner = 0
    max_time_outer = 0
    max_time_inner = 0
    
    for label_name, _ in label_map.items():
        word_dir = os.path.join(split_dir, "transcripts", label_name)
        time_dir = os.path.join(split_dir, "time", label_name)
        
        # We need a list of files to iterate through. Assume wav directory is the master list.
        wav_dir = os.path.join(split_dir, "wav", label_name)
        if not os.path.exists(wav_dir):
            continue
            
        file_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith(".mp3")])
        
        for file_id in file_ids:
            if load_words:
                word_file_pkl = os.path.join(word_dir, f"{file_id}.pkl")
                if os.path.exists(word_file_pkl):
                    with open(word_file_pkl, "rb") as f:
                        data = pickle.load(f)
                    
                    # Flatten the nested list to get the max length of the final flattened list
                    flattened_words = [w for sublist in data for w in sublist if isinstance(sublist, list) and w is not None]
                    max_word_outer = max(max_word_outer, len(flattened_words))
            
            if load_times:
                time_file_pkl = os.path.join(time_dir, f"{file_id}.pkl")
                if os.path.exists(time_file_pkl):
                    with open(time_file_pkl, "rb") as f:
                        data = pickle.load(f)

                    # Flatten the nested list to get the max length
                    flattened_times = [t for sublist in data for t in sublist if isinstance(sublist, list)]
                    max_time_outer = max(max_time_outer, len(flattened_times))

    # --- Second Pass: Load, flatten, pad, and stack the data ---
    for label_name, label_val in label_map.items():
        wav_dir = os.path.join(split_dir, "wav", label_name)
        if not os.path.exists(wav_dir):
            raise FileNotFoundError(f"Missing wav directory: {wav_dir}")
            
        file_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith(".mp3")])
        
        for file_id in file_ids:
            labels.append(label_val)

            if load_audio:
                audio_file = os.path.join(split_dir, "wav", label_name, f"{file_id}.mp3")
                audio_data.append(load_audio_file(audio_file, target_length))

            if load_words:
                word_dir = os.path.join(split_dir, "transcripts", label_name)
                word_file_pkl = os.path.join(word_dir, f"{file_id}.pkl")
                if os.path.exists(word_file_pkl):
                    with open(word_file_pkl, "rb") as f:
                        data = pickle.load(f)
                    
                    # Flatten the nested list
                    flattened_words = [w for sublist in data for w in sublist if isinstance(sublist, list) and w is not None]
                    all_word_lists.append(flattened_words)

            if load_times:
                time_dir = os.path.join(split_dir, "timestamps", label_name)
                time_file_pkl = os.path.join(time_dir, f"{file_id}.pkl")
                if os.path.exists(time_file_pkl):
                    with open(time_file_pkl, "rb") as f:
                        data = pickle.load(f)

                    # Flatten the nested list
                    flattened_times = [t for sublist in data for t in sublist if isinstance(sublist, list) and len(t) == 2]
                    all_time_lists.append(flattened_times)
            
    # Convert lists to numpy arrays for consistency
    audio_data = np.stack(audio_data) if audio_data else None
    
    # We will pad these sequences later on, after converting words to IDs
    word_data = all_word_lists
    time_data = all_time_lists
    labels = np.array(labels)
    
    return audio_data, word_data, time_data, labels

# utils.py

import numpy as np

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