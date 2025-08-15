import os
import shutil
import random
from pathlib import Path

# -----------------------
# CONFIGURATION
# -----------------------
PITT_DIR = Path("/path/to/pitt")   # path to your original Pitt Corpus folder
OUTPUT_DIR = Path("pitt_split")    # new clean split folder
METADATA_FILE = Path("/path/to/participant_metadata.csv")  # optional

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)

# -----------------------
# STEP 1: Read participant list
# -----------------------
# If no metadata, just take folder names as participants
participants = [p for p in PITT_DIR.iterdir() if p.is_dir()]

# Optional: if you have a CSV with participant -> label mapping, read it
labels = {}
if METADATA_FILE.exists():
    import csv
    with open(METADATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["participant_id"]
            diag = row["diagnosis"]  # e.g., "control" or "dementia"
            labels[pid] = diag

# -----------------------
# STEP 2: Split participants
# -----------------------
random.shuffle(participants)
n_total = len(participants)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_participants = participants[:n_train]
val_participants = participants[n_train:n_train+n_val]
test_participants = participants[n_train+n_val:]

splits = {
    "train": train_participants,
    "val": val_participants,
    "test": test_participants
}

# -----------------------
# STEP 3: Copy files
# -----------------------
for split_name, part_list in splits.items():
    audio_out = OUTPUT_DIR / split_name / "audio"
    trans_out = OUTPUT_DIR / split_name / "transcripts"
    audio_out.mkdir(parents=True, exist_ok=True)
    trans_out.mkdir(parents=True, exist_ok=True)

    for participant in part_list:
        for file in participant.iterdir():
            if file.suffix == ".wav":
                shutil.copy(file, audio_out / file.name)
            elif file.suffix == ".cha":
                shutil.copy(file, trans_out / file.name)

print(f"✅ Done! Split saved to {OUTPUT_DIR}")
