import os
import shutil
import random

# Define the top-level folders for your dataset categories
SOURCE_FOLDERS = ["control", "dementia"]
# The destination folder for the split dataset
DEST_FOLDER = "pitt_split"
# Split ratios
TRAIN_RATIO = 4/5
VAL_RATIO_OF_TRAIN = 4/5

def get_all_files():
    all_files = []
    for category in SOURCE_FOLDERS:
        source_dir = os.path.join("pitt_split1/transcripts", category)
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Directory not found: {source_dir}. Skipping.")

        for fname in os.listdir(source_dir):
            if fname.endswith(".pkl"):
                # Get the filename without the extension
                base_name = os.path.splitext(fname)[0]
                all_files.append({"filename": base_name, "category": category})
    return all_files

def organize_files(file_list):
    # Shuffle to ensure a random split each time
    random.shuffle(file_list)
    
    # First split: 4:1 train to test
    total_count = len(file_list)
    train_end = int(total_count * TRAIN_RATIO)
    temp_train_set = file_list[:train_end]
    test_set = file_list[train_end:]

    # Second split: 4:1 train to val on the training set
    temp_train_count = len(temp_train_set)
    train_end_final = int(temp_train_count * VAL_RATIO_OF_TRAIN)
    train_set = temp_train_set[:train_end_final]
    val_set = temp_train_set[train_end_final:]
    
    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    # Create destination directories
    for split_name in splits:
        for data_type in ["wav", "transcripts", "time"]:
            for category in SOURCE_FOLDERS:
                os.makedirs(os.path.join(DEST_FOLDER, split_name, data_type, category), exist_ok=True)
    
    # Copy files
    for split_name, files_to_copy in splits.items():
        for file_info in files_to_copy:
            base_name = file_info["filename"]
            category = file_info["category"]
            
            # The paths for the source files
            audio_source = os.path.join("pitt_split1", "wav", category, base_name + ".mp3") 
            transcripts_source = os.path.join("pitt_split1", "transcripts", category, base_name + ".pkl")
            timestamps_source = os.path.join("pitt_split1", "time", category, base_name + ".pkl")
            
            # The paths for the destination files, now with the category subfolder
            audio_dest = os.path.join(DEST_FOLDER, split_name, "wav", category, f"{base_name}.mp3")
            transcripts_dest = os.path.join(DEST_FOLDER, split_name, "transcripts", category, f"{base_name}.pkl")
            timestamps_dest = os.path.join(DEST_FOLDER, split_name, "time", category, f"{base_name}.pkl")
            
            # Copy audio files
            if os.path.exists(audio_source):
                shutil.copy(audio_source, audio_dest)
            else:
                raise FileNotFoundError(f"X Audio file not found: {audio_source}")
                
            # Copy transcripts files
            if os.path.exists(transcripts_source):
                shutil.copy(transcripts_source, transcripts_dest)
            else:
                raise FileNotFoundError(f"X Transcript file not found: {transcripts_source}")
            
            # Copy timestamps files
            if os.path.exists(timestamps_source):
                shutil.copy(timestamps_source, timestamps_dest)
            else:
                raise FileNotFoundError(f"X Timestamp file not found: {timestamps_source}")

if __name__ == "__main__":
    all_files = get_all_files()
    organize_files(all_files)
