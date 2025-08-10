import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import random

# ===== CONFIG =====
base_dir = r'D:\Uni\thesis1000\thesis1000\diagnosis'
splits = ["train", "test"]
classes = ["ad", "cn"]

sr = 16000       # sample rate
duration = 1.0   # seconds
seq_len = 50     # tokens per transcript
time_feat_dim = 2
vocab = {f"word{i}": i for i in range(1, 101)}

# ===== FOLDER CREATION =====
for split in splits:
    for c in classes:
        os.makedirs(os.path.join(base_dir, split, "audio", c), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "specto", c), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "trans", c), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "time", c), exist_ok=True)

# ===== DATA GENERATION =====
def generate_dummy_audio():
    """Generate 1 second of random audio at sr Hz."""
    return np.random.randn(int(sr)).astype(np.float32)

def save_spectrogram(audio, path):
    """Generate and save a spectrogram image from audio."""
    plt.specgram(audio, Fs=sr)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_transcript():
    """Generate a random transcript of words from vocab."""
    return " ".join(random.choice(list(vocab.keys())) for _ in range(seq_len))

def generate_time_features():
    """Generate dummy time-aligned features (seq_len, 2)."""
    return np.random.randn(seq_len, time_feat_dim).astype(np.float32)

# ===== MAIN LOOP =====
for split in splits:
    for c in classes:
        for i in range(3):  # 3 samples per class per split
            uid = f"{split}_{c}_{i}"
            
            # 1. Audio
            audio = generate_dummy_audio()
            audio_path = os.path.join(base_dir, split, "audio", c, f"{uid}.wav")
            sf.write(audio_path, audio, sr)
            
            # 2. Spectrogram
            spectro_path = os.path.join(base_dir, split, "specto", c, f"{uid}.png")
            save_spectrogram(audio, spectro_path)
            
            # 3. Transcript
            transcript = generate_transcript()
            trans_path = os.path.join(base_dir, split, "trans", c, f"{uid}.txt")
            with open(trans_path, "w") as f:
                f.write(transcript + "\n")
            
            # 4. Time features
            time_feat = generate_time_features()
            time_path = os.path.join(base_dir, split, "time", c, f"{uid}.npy")
            np.save(time_path, time_feat)

print("✅ Dummy multimodal dataset created successfully.")
