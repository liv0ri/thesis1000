# generate_dummy_pitt_multimodal_pkl.py
import os
import numpy as np
import soundfile as sf
import pickle

# -----------------
# CONFIG
# -----------------
splits = ["train", "val", "test"]
base_dir = "pitt_split"
split_counts = {
    "train": 10,
    "val": 3,
    "test": 3
}

sr = 16000          # sample rate for audio
audio_length = sr   # 1 second of audio
seq_len = 50        # number of tokens per sample
vocab_size = 100    # vocabulary size
time_features = 2   # start & end times



# -----------------
# GENERATE
# -----------------
os.makedirs(base_dir, exist_ok=True)

for split in splits:
    audio_dir = os.path.join(base_dir, split, "audio")
    word_dir = os.path.join(base_dir, split, "words")
    time_dir = os.path.join(base_dir, split, "times") 
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(word_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)

    labels = []

    for i in range(split_counts[split]):
        # file base name
        base_name = f"{split}_{i:03d}"

        # --- audio ---
        audio_data = np.random.randn(audio_length).astype(np.float32)
        sf.write(os.path.join(audio_dir, base_name + ".wav"), audio_data, sr)

        # --- words ---
        # Random IDs between 1 and vocab_size (inclusive)
        word_ids = np.random.randint(1, vocab_size + 1, size=(seq_len,), dtype=np.int32)
        with open(os.path.join(word_dir, base_name + ".pkl"), "wb") as f:
            pickle.dump(word_ids, f)

        # --- times ---
        times = np.random.rand(seq_len, time_features).astype(np.float32)
        with open(os.path.join(time_dir, base_name + ".pkl"), "wb") as f:
            pickle.dump(times, f)

        # --- label ---
        label = np.random.randint(0, 2)
        labels.append(label)

    # save labels.pkl
    with open(os.path.join(base_dir, split, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

print(f"✅ Dummy multimodal Pitt-like dataset (with .pkl) created in '{base_dir}'")

# -----------------
# CREATE VOCAB
# -----------------
# IDs go from 1 to vocab_size (0 reserved for <PAD>/<UNK>)
vocab = {f"word{i}": i for i in range(1, vocab_size + 1)}
with open(os.path.join(base_dir, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

# Create embeddings directly from vocab
embedding_dim = 300
word2vec_vectors = {word: np.random.rand(embedding_dim).astype(np.float32) for word in vocab}

# Save embeddings to a file
os.makedirs("embeddings", exist_ok=True)
with open("embeddings/word2vec_vectors.pkl", "wb") as f:
    pickle.dump(word2vec_vectors, f)

print(f"📦 vocab size: {len(vocab)} (IDs 1..{vocab_size}, 0 reserved for PAD/UNK)")
