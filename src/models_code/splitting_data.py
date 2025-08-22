import os
import shutil
import random

# --- User-defined variables ---
SOURCE_FOLDERS = ["control", "dementia"]
DEST_FOLDER = "pitt_split"

def get_files_by_category():
    """Gathers all .cha and .wav files and categorizes them."""
    all_files = {"control": [], "dementia": []}
    for folder in SOURCE_FOLDERS:
        for fname in os.listdir(folder):
            if fname.endswith(".cha"):
                category = os.path.basename(os.path.normpath(folder))
                all_files[category].append(os.path.splitext(fname)[0])
    return all_files

def split_and_copy(files_by_category):
    """Splits filenames based on 4:1 ratio and copies files."""
    for category, file_list in files_by_category.items():
        random.shuffle(file_list)
        
        # First split: 4:1 train to test (80% train, 20% test)
        total_count = len(file_list)
        train_end = int(total_count * 4/5)
        
        temp_train_files = file_list[:train_end]
        test_files = file_list[train_end:]
        
        # Second split: 4:1 train to val on the training set (80% train, 20% val)
        random.shuffle(temp_train_files)
        temp_train_count = len(temp_train_files)
        train_end_final = int(temp_train_count * 4/5)
        
        train_files = temp_train_files[:train_end_final]
        val_files = temp_train_files[train_end_final:]

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        # Copy files to the new structure
        for split_name, filenames in splits.items():
            for filename in filenames:
                # Source paths for .cha and .wav
                cha_source_path = os.path.join(category, filename + ".cha")
                wav_source_path = os.path.join(category, filename + ".wav")

                # Destination paths
                dest_path_cha = os.path.join(DEST_FOLDER, split_name, category, filename + ".cha")
                dest_path_wav = os.path.join(DEST_FOLDER, split_name, category, filename + ".wav")
                
                # Create destination directories if they don't exist
                os.makedirs(os.path.dirname(dest_path_cha), exist_ok=True)
                os.makedirs(os.path.dirname(dest_path_wav), exist_ok=True)
                
                # Copy the files
                if os.path.exists(cha_source_path):
                    shutil.copy(cha_source_path, dest_path_cha)
                if os.path.exists(wav_source_path):
                    shutil.copy(wav_source_path, dest_path_wav)
                print(f"Copied {filename} to {split_name}/{category}")

if __name__ == "__main__":
    files = get_files_by_category()
    split_and_copy(files)