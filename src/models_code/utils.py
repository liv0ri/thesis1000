import os
import numpy as np
import pickle
import soundfile as sf

def load_audio_file(file_path, target_length=16000):
    """
    Loads an audio file and pads/truncates it to a target length.
    """
    audio, sr = sf.read(file_path)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        # Pad with zeros to meet the target length
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))
    return audio.astype(np.float32)

def load_split(root_dir, target_length=16000, load_audio=True, load_words=True, load_times=True):
    """
    Loads data from a single split (e.g., 'train', 'val', or 'test')
    of the Pitt dataset, considering 'control' and 'dementia' subfolders.

    Args:
        root_dir (str): Path to the specific split directory (e.g., 'pitt_split/train').
        target_length (int): Padding/truncation length for audio samples.
        load_audio (bool): Whether to load audio data.
        load_words (bool): Whether to load word transcript data.
        load_times (bool): Whether to load timestamp data.

    Returns:
        tuple: (audio_data, word_data, time_data, labels) as stacked numpy arrays.
    """

    # The directory where the data for a specific split is located
    split_dir = root_dir
    audio_data, word_data, time_data, labels = [], [], [], []

    # Map category names to numerical labels
    label_map = {"control": 0, "dementia": 1}

    # Iterate through the 'control' and 'dementia' subdirectories
    for label_name, label_val in label_map.items():
        # Construct the path to the directory for this category's audio files
        wav_dir = os.path.join(split_dir, "wav", label_name)
        
        if not os.path.exists(wav_dir):
            # print(f"Directory {wav_dir} does not exist. Skipping this category.")
            continue

        # Get a sorted list of all file IDs (base names without extension)
        file_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith(".mp3")])

        for file_id in file_ids:
            # Append the label for this file to the labels list
            labels.append(label_val)

            # Load Audio data
            if load_audio:
                audio_file = os.path.join(split_dir, "wav", label_name, f"{file_id}.mp3")
                if os.path.exists(audio_file):
                    audio_data.append(load_audio_file(audio_file, target_length))
                else:
                    raise ValueError(f"Missing audio file: {audio_file}. Using placeholder.")

            # Load Words (transcripts)
            if load_words:
                word_dir = os.path.join(split_dir, "transcripts", label_name)
                word_file_pkl = os.path.join(word_dir, f"{file_id}.pkl")
                if os.path.exists(word_file_pkl):
                    with open(word_file_pkl, "rb") as f:
                        word_data.append(pickle.load(f))
                else:
                    # Append a placeholder if the file is missing
                    raise ValueError(f"Missing word file: {word_file_pkl}. Using placeholder.")

            # Load Times
            if load_times:
                time_dir = os.path.join(split_dir, "time", label_name)
                time_file_pkl = os.path.join(time_dir, f"{file_id}.pkl")
                if os.path.exists(time_file_pkl):
                    with open(time_file_pkl, "rb") as f:
                        time_data.append(pickle.load(f))
                else:
                    # Append a placeholder if the file is missing
                    raise ValueError(f"Missing time file: {time_file_pkl}. Using placeholder.")
                
    # Convert lists to numpy arrays for consistency
    audio_data = np.stack(audio_data) if audio_data else None
    word_data = np.stack(word_data) if word_data else None
    time_data = np.stack(time_data) if time_data else None
    labels = np.array(labels)
    # print(f"Loaded {len(labels)} samples from {root_dir}")

    return audio_data, word_data, time_data, labels
