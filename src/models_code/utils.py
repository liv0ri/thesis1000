import os
import numpy as np
import pickle
import soundfile as sf
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

def _pad_nested_list(nested_list, max_outer_len, max_inner_len, pad_value=None):
    # Pad inner lists
    padded_inner_list = []
    for inner_list in nested_list:
        padded_inner = inner_list + [pad_value] * (max_inner_len - len(inner_list))
        padded_inner_list.append(padded_inner)
    
    # Pad outer list
    padded_outer_list = padded_inner_list + [[pad_value] * max_inner_len] * (max_outer_len - len(padded_inner_list))
    
    return padded_outer_list

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
                        max_word_outer = max(max_word_outer, len(data))
                        for inner_list in data:
                            max_word_inner = max(max_word_inner, len(inner_list))
            
            if load_times:
                time_file_pkl = os.path.join(time_dir, f"{file_id}.pkl")
                if os.path.exists(time_file_pkl):
                    with open(time_file_pkl, "rb") as f:
                        data = pickle.load(f)
                        max_time_outer = max(max_time_outer, len(data))
                        for inner_list in data:
                            max_time_inner = max(max_time_inner, len(inner_list))

    # --- Second Pass: Load, pad, and stack the data ---
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
                        padded_data = _pad_nested_list(data, max_word_outer, max_word_inner, pad_value=None)
                        all_word_lists.append(padded_data)
                else:
                    raise ValueError(f"Missing word file: {word_file_pkl}. Using placeholder.")


            if load_times:
                time_dir = os.path.join(split_dir, "time", label_name)
                time_file_pkl = os.path.join(time_dir, f"{file_id}.pkl")
                if os.path.exists(time_file_pkl):
                    with open(time_file_pkl, "rb") as f:
                        data = pickle.load(f)
                        padded_data = _pad_nested_list(data, max_time_outer, max_time_inner, pad_value=[0, 0])
                        all_time_lists.append(padded_data)
                else:
                    raise ValueError(f"Missing time file: {time_file_pkl}. Using placeholder.")
                
    # Convert lists to numpy arrays for consistency
    audio_data = np.stack(audio_data) if audio_data else None
    word_data = np.array(all_word_lists, dtype=object) if all_word_lists else None
    time_data = np.array(all_time_lists, dtype=object) if all_time_lists else None
    labels = np.array(labels)
    
    print(f"Loaded {len(labels)} samples from {root_dir}")

    return audio_data, word_data, time_data, labels

def pad_sequences_and_times_np(word_sequences=None, time_sequences=None, maxlen=100):
    num_samples = 0
    if word_sequences is not None:
        num_samples = len(word_sequences)
    elif time_sequences is not None:
        num_samples = len(time_sequences)
    else:
        raise ValueError("At least one of word_sequences or time_sequences must be provided.")

    padded_words_np = None
    padded_times_np = None

    if word_sequences is not None:
        padded_words_np = np.zeros((num_samples, maxlen), dtype="int32")

    if time_sequences is not None:
        padded_times_np = np.zeros((num_samples, maxlen, 2), dtype="float32")

    for i in range(num_samples):
        if word_sequences is not None:
            words = word_sequences[i]
            cleaned_words = [w for w in words if w is not None and isinstance(w, int)]
            seq_len = min(len(cleaned_words), maxlen)
            if seq_len > 0:
                padded_words_np[i, :seq_len] = cleaned_words[:seq_len]

        # --- times ---
        if time_sequences is not None:
            times = time_sequences[i]

            # ensure each element is a flat [float, float]
            cleaned_times = []
            for t in times:
                if t is None:
                    continue
                if isinstance(t, (list, tuple)) and len(t) == 1 and isinstance(t[0], (list, tuple)):
                    t = t[0]
                if isinstance(t, (list, tuple)) and len(t) == 2:
                    try:
                        cleaned_times.append([float(t[0]), float(t[1])])
                    except (TypeError, ValueError):
                        continue

            seq_len = min(len(cleaned_times), maxlen)
            if seq_len > 0:
                padded_times_np[i, :seq_len, :] = np.array(cleaned_times[:seq_len], dtype="float32")

    return padded_words_np, padded_times_np
