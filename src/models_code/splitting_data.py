import os
import pickle
import pylangacq
from sklearn.model_selection import train_test_split
from pydub import AudioSegment

# ---------------- CONFIG ----------------
INPUT_BASE_PATH = "pitt_split1"   # where cha_files/ and wav/ are
OUTPUT_BASE_PATH = "pitt_split"   # final dataset folder

TEST_RATIO = 0.2
VAL_RATIO = 0.2

INPUT_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "cha_files", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "cha_files", "dementia"),
}

AUDIO_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "wav", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "wav", "dementia"),
}

# ---------------- HELPERS ----------------
def extract_utterances(cha_path, label):
    """Extract utterances from .cha file with robust timestamp handling"""
    reader = pylangacq.read_chat(cha_path)
    utterances = reader.utterances()

    data_points = []
    for utt in utterances:
        words = [tok.word for tok in utt.tokens if tok.word and tok.word.isalpha()]
        if not utt.time_marks or not words:
            continue

        start, end = utt.time_marks

        # Skip None
        if start is None or end is None:
            continue

        # Swap if end < start
        if end < start:
            start, end = end, start

        # Skip zero-length or extremely short utterances
        if end - start < 1:  # at least 1 ms
            continue

        # Uniform distribution of word timestamps
        word_times = [(start + i*(end-start)/len(words), start + (i+1)*(end-start)/len(words))
                      for i in range(len(words))]

        data_points.append({
            "words": words,
            "word_times": word_times,
            "start_time": start,
            "end_time": end,
            "label": label,
            "source_file": os.path.basename(cha_path)
        })

    return data_points

def slice_audio_mp3(audio_path, start_time, end_time):
    """Load MP3 audio with pydub and slice between start/end seconds (with clipping)"""
    if not os.path.exists(audio_path):
        print(f"Audio not found: {audio_path}")
        return None

    audio = AudioSegment.from_file(audio_path)
    audio_len_ms = len(audio)

    # Convert milliseconds in .cha to pydub ms
    start_ms = int(start_time)
    end_ms = int(end_time)

    # Clip to audio length
    start_ms = max(0, min(start_ms, audio_len_ms-1))
    end_ms = max(0, min(end_ms, audio_len_ms))

    if end_ms <= start_ms:
        print(f"Warning: empty snippet {audio_path}")
        return None

    snippet = audio[start_ms:end_ms]
    return snippet

def save_datapoint(dp, audio_folder, out_base, idx):
    """Save transcripts, timestamps, and MP3 audio snippets"""
    label = dp["label"]
    base_name = os.path.splitext(dp["source_file"])[0] + f"_{idx}"

    # Save transcripts
    transcript_dir = os.path.join(out_base, "transcripts", label)
    os.makedirs(transcript_dir, exist_ok=True)
    with open(os.path.join(transcript_dir, base_name + ".pkl"), "wb") as f:
        pickle.dump(dp.get("words", []), f)

    # Save timestamps
    timestamps_dir = os.path.join(out_base, "timestamps", label)
    os.makedirs(timestamps_dir, exist_ok=True)
    with open(os.path.join(timestamps_dir, base_name + ".pkl"), "wb") as f:
        pickle.dump(dp.get("word_times", []), f)

    # Save audio snippet as MP3
    audio_dir = os.path.join(out_base, "wav", label)
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_folder[label], dp["source_file"].replace(".cha", ".mp3"))
    if not os.path.exists(audio_path):
        print(f"Warning: audio not found {audio_path}")
        return

    snippet = slice_audio_mp3(audio_path, dp["start_time"], dp["end_time"])
    if snippet is None:
        print(f"Warning: empty audio snippet for {dp['source_file']}")
        return

    snippet.export(os.path.join(audio_dir, base_name + ".mp3"), format="mp3")

# ---------------- MAIN ----------------
def main():
    all_data = []

    # Extract utterances
    for label, folder in INPUT_FOLDERS.items():
        for fname in os.listdir(folder):
            if fname.endswith(".cha"):
                cha_path = os.path.join(folder, fname)
                utterances = extract_utterances(cha_path, label)
                all_data.extend(utterances)

    print(f"Total utterances extracted: {len(all_data)}")

    # Stratified split
    labels = [1 if d["label"] == "dementia" else 0 for d in all_data]
    train, test = train_test_split(all_data, test_size=TEST_RATIO,
                                   stratify=labels, random_state=42)
    train, val = train_test_split(
        train, test_size=VAL_RATIO,
        stratify=[1 if d["label"]=="dementia" else 0 for d in train],
        random_state=42
    )

    splits = {"train": train, "val": val, "test": test}

    # Save split data
    for split_name, split_data in splits.items():
        print(f"Saving {len(split_data)} utterances to {split_name}...")
        out_base = os.path.join(OUTPUT_BASE_PATH, split_name)
        for idx, dp in enumerate(split_data):
            save_datapoint(dp, AUDIO_FOLDERS, out_base, idx)

    print("✅ Finished preprocessing & splitting.")

if __name__ == "__main__":
    main()
