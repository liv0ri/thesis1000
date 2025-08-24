import os
import pylangacq
import pickle

# --- User-defined variables ---
INPUT_BASE_PATH = "pitt_split1"
OUTPUT_BASE_PATH = "pitt_split1"

def process_cha_file(filepath):
    try:
        reader = pylangacq.read_chat(filepath)
        utterances = reader.utterances()

        transcripts = []
        timestamps = []

        for utt in utterances:
            # Extract words, filtering out any non-alphabetic tokens
            words = [tok.word for tok in utt.tokens if tok.word and tok.word.isalpha()]
            
            # Check for time marks
            if not utt.time_marks:
                continue

            start, end = utt.time_marks
            
            if not words or start is None or end is None:
                continue

            # Calculate the duration of the utterance, ensuring it's not zero
            duration = max(1e-6, end - start)
            word_times = []
            n_words = len(words)

            # Distribute time evenly among words
            for i, w in enumerate(words):
                # Calculate start and end time as a fraction of the utterance duration
                s = (start + i * duration / n_words - start) / duration
                e = (start + (i + 1) * duration / n_words - start) / duration
                word_times.append([s, e])

            transcripts.append(words)
            timestamps.append(word_times)
    except Exception as e:
        raise ValueError(f"Error processing file {filepath}: {e}")

    return transcripts, timestamps

def save_pickle(data, filepath):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def pad_list_of_lists(data, max_outer_len, max_inner_len):
    padded_data = []
    # Pad the outer list
    for outer_item in data:
        # Pad the inner lists
        padded_inner = [inner_item + [None] * (max_inner_len - len(inner_item)) for inner_item in outer_item]
        # Pad the outer list itself
        padded_outer = padded_inner + [[]] * (max_outer_len - len(padded_inner))
        padded_data.append(padded_outer)
        
    return padded_data

def main():
    # Define input folders based on the INPUT_BASE_PATH
    INPUT_FOLDERS = [
        os.path.join(INPUT_BASE_PATH, "cha_files", "control"),
        os.path.join(INPUT_BASE_PATH, "cha_files", "dementia")
    ]
    
    TRANSCRIPT_FOLDER = os.path.join(OUTPUT_BASE_PATH, "transcripts")
    TIMESTAMP_FOLDER = os.path.join(OUTPUT_BASE_PATH, "time")
    
    all_data = []

    # print("Beginning first pass: processing all files to find max lengths...")
    for folder in INPUT_FOLDERS:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Skipping folder: {folder} not found.")
        
        for fname in os.listdir(folder):
            if fname.endswith(".cha"):
                cha_path = os.path.join(folder, fname)
                transcripts, timestamps = process_cha_file(cha_path)
                
                if transcripts:
                    all_data.append({
                        "file_path": cha_path,
                        "folder_name": os.path.basename(os.path.normpath(folder)),
                        "transcripts": transcripts,
                        "timestamps": timestamps,
                    })

    # Calculate maximum lengths for padding
    max_transcript_outer_len = max(len(d["transcripts"]) for d in all_data) if all_data else 0
    max_timestamp_outer_len = max(len(d["timestamps"]) for d in all_data) if all_data else 0
    
    max_transcript_inner_len = 0
    max_timestamp_inner_len = 0
    
    for d in all_data:
        for t in d["transcripts"]:
            max_transcript_inner_len = max(max_transcript_inner_len, len(t))
        for t in d["timestamps"]:
            max_timestamp_inner_len = max(max_timestamp_inner_len, len(t))

    # print(f"\nMax utterance count: {max_transcript_outer_len}")
    # print(f"Max words per utterance: {max_transcript_inner_len}")
    
    for data_item in all_data:
        file_path = data_item["file_path"]
        folder_name = data_item["folder_name"]
        
        # Pad the transcripts and timestamps - take the first element from a list of length 1
        padded_transcripts = pad_list_of_lists([data_item["transcripts"]], max_transcript_outer_len, max_transcript_inner_len)[0]
        padded_timestamps = pad_list_of_lists([data_item["timestamps"]], max_timestamp_outer_len, max_timestamp_inner_len)[0]

        base = os.path.splitext(os.path.basename(file_path))[0]
        
        transcript_dir = os.path.join(TRANSCRIPT_FOLDER, folder_name)
        timestamp_dir = os.path.join(TIMESTAMP_FOLDER, folder_name)
        os.makedirs(transcript_dir, exist_ok=True)
        os.makedirs(timestamp_dir, exist_ok=True)

        save_pickle(padded_transcripts, os.path.join(transcript_dir, base + ".pkl"))
        save_pickle(padded_timestamps, os.path.join(timestamp_dir, base + ".pkl"))

if __name__ == "__main__":
    main()