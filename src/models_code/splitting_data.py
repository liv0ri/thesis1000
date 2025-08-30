import os
import pickle
import pylangacq
import numpy as np
from random import shuffle
import soundfile as sf
import warnings
from config import INPUT_BASE_PATH, PROCESSED_DATA_PATH, TARGET_AUDIO_LENGTH

INPUT_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "cha_files", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "cha_files", "dementia"),
}

AUDIO_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "wav", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "wav", "dementia"),
}

def extract_utterances(cha_path, label):
    reader = pylangacq.read_chat(cha_path)
    utterances = reader.utterances()
    data_points = []
    for utt in utterances:
        words = [tok.word for tok in utt.tokens if tok.word and tok.word.isalpha()]
        if not utt.time_marks or not words:
            continue
        start, end = utt.time_marks
        if start is None or end is None or end < start or end - start < 1:
            continue
        data_points.append({
            "words": words,
            "start": start,
            "end": end,
            "label": label,
            "source_file": os.path.basename(cha_path)
        })
    return data_points


def load_and_save_data(input_folders, audio_folders, output_path, target_length=TARGET_AUDIO_LENGTH):
    all_data_points = []
    for label, folder in input_folders.items():
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".cha"):
                cha_path = os.path.join(folder, fname)
                base_name = os.path.splitext(fname)[0]
                audio_path = os.path.join(audio_folders[label], base_name + ".mp3")
                
                try:
                    # Load the audio file once per transcript and get the sample rate
                    audio, sample_rate = sf.read(audio_path)
                except FileNotFoundError:
                    warnings.warn(f"Audio file not found for {cha_path}, skipping.")
                    continue
                
                # Check for multiple channels and convert to mono if necessary
                if audio.ndim > 1:
                    audio = audio[:, 0]
                
                utterances = extract_utterances(cha_path, label)
                
                for utt in utterances:
                    start_frame = int((utt['start'] / 1000) * sample_rate)
                    end_frame = int((utt['end'] / 1000) * sample_rate)

                    # Ensure valid frame indices
                    start_frame = max(0, start_frame)
                    end_frame = min(len(audio), end_frame)

                    if start_frame >= end_frame:
                        continue

                    # Extract the utterance segment
                    utt_audio = audio[start_frame:end_frame]

                    # Pad or truncate the segment to the target length (center padding)
                    if len(utt_audio) > target_length:
                        utt_audio = utt_audio[:target_length]
                    else:
                        pad_len = target_length - len(utt_audio)
                        left_pad = pad_len // 2
                        right_pad = pad_len - left_pad
                        utt_audio = np.pad(utt_audio, (left_pad, right_pad))
                    
                    utt['audio'] = utt_audio.astype(np.float32)
                    all_data_points.append(utt)

    print(f"Total utterances extracted: {len(all_data_points)}")
    shuffle(all_data_points)

    with open(output_path, "wb") as f:
        pickle.dump(all_data_points, f)

if __name__ == "__main__":
    load_and_save_data(INPUT_FOLDERS, AUDIO_FOLDERS, PROCESSED_DATA_PATH)