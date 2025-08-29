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
        word_times = [(start + i*(end-start)/len(words), start + (i+1)*(end-start)/len(words))
                      for i in range(len(words))]
        data_points.append({
            "words": words,
            "word_times": word_times,
            "label": label,
            "source_file": os.path.basename(cha_path)
        })
    return data_points

def load_audio_file(file_path, target_length=TARGET_AUDIO_LENGTH):
    try:
        audio, _ = sf.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if audio.ndim > 1:
        warnings.warn(f"Audio file has multiple channels. Using the first channel.")
        audio = audio[:, 0]
        
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))
        
    return audio.astype(np.float32)

def load_and_save_data(input_folders, audio_folders, output_path):
    all_data_points = []
    for label, folder in input_folders.items():
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".cha"):
                cha_path = os.path.join(folder, fname)
                base_name = os.path.splitext(fname)[0]
                audio_path = os.path.join(audio_folders[label], base_name + ".mp3") 
                
                audio = load_audio_file(audio_path)
                if audio is None:
                    raise FileNotFoundError(f"Audio file not found for {cha_path}")

                utterances = extract_utterances(cha_path, label)
                
                for utt in utterances:
                    utt['audio'] = audio
                    all_data_points.append(utt)
    
    print(f"Total utterances extracted: {len(all_data_points)}")

    shuffle(all_data_points)

    with open(output_path, "wb") as f:
        pickle.dump(all_data_points, f)
    
if __name__ == "__main__":
    load_and_save_data(INPUT_FOLDERS, AUDIO_FOLDERS, PROCESSED_DATA_PATH)