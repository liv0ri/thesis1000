import os
import numpy as np
import pickle
import soundfile as sf

def load_audio_file(file_path, target_length=16000):
    audio, sr = sf.read(file_path)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))
    return audio.astype(np.float32)

def load_split(split_dir, load_audio=True, load_words=True, load_times=True):
    # Args:
    #     split_dir (str): path to the split folder
    #     load_audio (bool): whether to load audio files
    #     load_words (bool): whether to load transcript word IDs
    #     load_times (bool): whether to load timestamps

    # Returns:
    #     tuple: (audio_data, word_data, time_data, labels)
    #            Missing modalities will be returned as None.

    split_name = os.path.basename(split_dir.rstrip("/\\"))
    audio_data, word_data, time_data = None, None, None

    # Load labels
    labels_path = os.path.join(split_dir, "labels.pkl")
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    num_samples = len(labels)

    # Audio
    if load_audio:
        audio_dir = os.path.join(split_dir, "audio")
        if os.path.exists(audio_dir):
            audio_data = [
                load_audio_file(os.path.join(audio_dir, f"{split_name}_{i:03d}.wav"))
                for i in range(num_samples)
            ]
            audio_data = np.stack(audio_data)

    # Words
    if load_words:
        words_dir = os.path.join(split_dir, "words")
        if os.path.exists(words_dir):
            word_data = []
            for i in range(num_samples):
                word_file_npy = os.path.join(words_dir, f"{split_name}_{i:03d}.npy")
                word_file_pkl = os.path.join(words_dir, f"{split_name}_{i:03d}.pkl")

                if os.path.exists(word_file_npy):
                    word_data.append(np.load(word_file_npy))
                elif os.path.exists(word_file_pkl):
                    with open(word_file_pkl, "rb") as f:
                        word_data.append(pickle.load(f))
            word_data = np.stack(word_data)

    # Times
    if load_times:
        time_dir = os.path.join(split_dir, "times")
        if os.path.exists(time_dir):
            time_data = []
            for i in range(num_samples):
                time_file_npy = os.path.join(time_dir, f"{split_name}_{i:03d}.npy")
                time_file_pkl = os.path.join(time_dir, f"{split_name}_{i:03d}.pkl")

                if os.path.exists(time_file_npy):
                    time_data.append(np.load(time_file_npy))
                elif os.path.exists(time_file_pkl):
                    with open(time_file_pkl, "rb") as f:
                        time_data.append(pickle.load(f))
            time_data = np.stack(time_data)

    return audio_data, word_data, time_data, np.array(labels)
