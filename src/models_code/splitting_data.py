import os
import pickle
import pylangacq
import numpy as np
from random import shuffle
import soundfile as sf
import warnings
import random
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
from config import INPUT_BASE_PATH, PROCESSED_DATA_PATH, TARGET_AUDIO_LENGTH

# nltk.download('wordnet')

INPUT_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "cha_files", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "cha_files", "dementia"),
}

AUDIO_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "wav", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "wav", "dementia"),
}

# Manual seeds (safe words you trust)
SEED_SYNONYM_DICT = {
    "get": ["obtain", "buy", "acquire"],
    "say": ["tell", "speak"],
    "talk": ["speak", "chat", "discuss"],
    "take": ["grab", "seize"],
    "help": ["assist", "aid"],
    "find": ["discover", "locate"],
    "start": ["begin", "commence"],
    "make": ["create", "build"],
    "go": ["move", "travel"],
}

def get_wordnet_synonyms(word, max_synonyms=5):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym.lower())
    return list(synonyms)[:max_synonyms]

def filter_common_words(words, min_freq=3.0):
    return [w for w in words if zipf_frequency(w, "en") >= min_freq and w.isalpha()]

def build_hybrid_synonym_dict(seed_dict, expand=True, max_synonyms=5, min_freq=3.0):
    hybrid_dict = {}
    for word, manual_syns in seed_dict.items():
        synonyms = set(manual_syns)
        if expand:
            wn_syns = get_wordnet_synonyms(word, max_synonyms=max_synonyms)
            synonyms.update(wn_syns)
        filtered_syns = filter_common_words(list(synonyms), min_freq=min_freq)
        if filtered_syns:
            hybrid_dict[word] = filtered_syns
    return hybrid_dict

def approximate_word_times(words, utt_start_ms, utt_end_ms):
    duration = utt_end_ms - utt_start_ms  # in ms
    step = duration / len(words)
    word_times = []
    for i in range(len(words)):
        s = i * step
        e = (i + 1) * step
        word_times.append((s, e))
    return word_times


def augment_text(words, replace_probability=0.1):
    augmented_words = []
    for word in words:
        if random.random() < replace_probability and word.lower() in SYNONYM_DICT:
            synonyms = SYNONYM_DICT[word.lower()]
            augmented_words.append(random.choice(synonyms))
        else:
            augmented_words.append(word)
    return augmented_words

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

        # Assume start and end are already in milliseconds
        start_ms = start
        end_ms = end

        word_times = approximate_word_times(words, start_ms, end_ms)

        data_points.append({
            "words": words,
            "word_times": word_times,  # per-word timestamps in ms
            "start": start_ms,         # utterance-level start in ms
            "end": end_ms,             # utterance-level end in ms
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
                    utt['source_type'] = 'original'
                    all_data_points.append(utt.copy())
                    augmented_text_utt = utt.copy()
                    augmented_text_utt['words'] = augment_text(utt['words'])
                    augmented_text_utt['source_type'] = 'text_augmented'
                    all_data_points.append(augmented_text_utt)

    print(f"Total utterances extracted: {len(all_data_points)}")
    shuffle(all_data_points)

    with open(output_path, "wb") as f:
        pickle.dump(all_data_points, f)

if __name__ == "__main__":
    SYNONYM_DICT = build_hybrid_synonym_dict(SEED_SYNONYM_DICT, expand=True, max_synonyms=8, min_freq=3.5)
    load_and_save_data(INPUT_FOLDERS, AUDIO_FOLDERS, PROCESSED_DATA_PATH)