import os
import pylangacq
import pickle

# --- User-defined variables ---
INPUT_BASE_PATH = "pitt_split1"
OUTPUT_BASE_PATH = "pitt_split1"

def process_cha_file(filepath):
    """
    Extracts words and timestamps from a .cha file.

    Args:
        filepath (str): The full path to the .cha file.

    Returns:
        tuple: A tuple containing a list of transcripts and a list of timestamps.
               Returns ([], []) if the file cannot be processed.
    """
    try:
        reader = pylangacq.read_chat(filepath)
        utterances = reader.utterances()

        transcripts = []
        timestamps = []

        for utt in utterances:
            # Extract words, filtering out any non-alphabetic tokens.
            words = [tok.word for tok in utt.tokens if tok.word and tok.word.isalpha()]
            
            # Check for time marks
            if not utt.time_marks:
                continue

            start, end = utt.time_marks
            
            if not words or start is None or end is None:
                continue

            # Calculate the duration of the utterance, ensuring it's not zero.
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
        print(f"Error processing file {filepath}: {e}")
        return [], []

    return transcripts, timestamps

def save_pickle(data, filepath):
    """
    Saves data to a pickle file.

    Args:
        data: The data to save.
        filepath (str): The path where the file will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def main():
    """
    Main function to process all .cha files and save the output.
    """
    # Define input folders based on the INPUT_BASE_PATH
    INPUT_FOLDERS = [
        os.path.join(INPUT_BASE_PATH, "cha_files", "control"),
        os.path.join(INPUT_BASE_PATH, "cha_files", "dementia")
    ]
    
    # Define output folder paths based on OUTPUT_BASE_PATH
    TRANSCRIPT_FOLDER = os.path.join(OUTPUT_BASE_PATH, "transcripts")
    TIMESTAMP_FOLDER = os.path.join(OUTPUT_BASE_PATH, "time")
    
    # Create output directories if they don't exist
    os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
    os.makedirs(TIMESTAMP_FOLDER, exist_ok=True)

    for folder in INPUT_FOLDERS:
        # Get the name of the last directory in the path (e.g., 'control' or 'dementia')
        folder_name = os.path.basename(os.path.normpath(folder))
        
        # Check if the folder exists before listing its contents
        if not os.path.exists(folder):
            print(f"Skipping folder: {folder} not found.")
            continue
            
        for fname in os.listdir(folder):
            if fname.endswith(".cha"):
                cha_path = os.path.join(folder, fname)
                transcripts, timestamps = process_cha_file(cha_path)
                
                if not transcripts:
                    print(f"Warning: No valid data extracted from {fname}")
                    continue
                
                # Create output subdirectories based on the input folder name
                transcript_dir = os.path.join(TRANSCRIPT_FOLDER, folder_name)
                timestamp_dir = os.path.join(TIMESTAMP_FOLDER, folder_name)
                os.makedirs(transcript_dir, exist_ok=True)
                os.makedirs(timestamp_dir, exist_ok=True)

                base = os.path.splitext(fname)[0]
                
                # Save to .pkl files in the respective subdirectories
                save_pickle(transcripts, os.path.join(transcript_dir, base + ".pkl"))
                save_pickle(timestamps, os.path.join(timestamp_dir, base + ".pkl"))

                print(f"Processed {fname} from {folder_name} → transcripts & timestamps saved")

if __name__ == "__main__":
    main()
