import os
from PIL import Image
import numpy as np

base_dir = r'D:\Uni\thesis1000\thesis1000\thesis1000\diagnosis\train'

folders = [
    'audio/ad',
    'audio/cn',
    'specto/ad',
    'specto/cn',
    'trans/ad',
    'trans/cn'
]

# Create folders
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)

# Create dummy .png images (e.g. 224x224 RGB)
dummy_image = Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))

for folder in ['audio/ad', 'audio/cn']:
    folder_path = os.path.join(base_dir, folder)
    for i in range(3):  # create 3 dummy images per folder
        dummy_image.save(os.path.join(folder_path, f'dummy_{i}.png'))

# Create dummy .txt transcript files matching image names
for folder in ['trans/ad', 'trans/cn']:
    folder_path = os.path.join(base_dir, folder)
    for i in range(3):
        with open(os.path.join(folder_path, f'dummy_{i}.txt'), 'w') as f:
            f.write("This is a dummy transcript for testing.\n")

# Create dummy spectrogram images (same as audio here for simplicity)
for folder in ['specto/ad', 'specto/cn']:
    folder_path = os.path.join(base_dir, folder)
    for i in range(3):
        dummy_image.save(os.path.join(folder_path, f'dummy_{i}.png'))

# Create test folders and files

test_audio_dir = r'D:\Uni\thesis1000\thesis1000\thesis1000\diagnosis\test-distaudio'
os.makedirs(test_audio_dir, exist_ok=True)
for i in range(3):
    dummy_image.save(os.path.join(test_audio_dir, f'dummy_{i}.png'))

test_specto_dir = r'D:\Uni\thesis1000\thesis1000\thesis1000\diagnosis\test-distspecto'
os.makedirs(test_specto_dir, exist_ok=True)
for i in range(3):
    dummy_image.save(os.path.join(test_specto_dir, f'dummy_{i}.png'))

test_trans_dir = r'D:\Uni\thesis1000\thesis1000\thesis1000\diagnosis\test-disttrans'
os.makedirs(test_trans_dir, exist_ok=True)
for i in range(3):
    with open(os.path.join(test_trans_dir, f'dummy_{i}.txt'), 'w') as f:
        f.write("This is a dummy transcript for testing.\n")

test_specto_txt_dir = r'D:\Uni\thesis1000\thesis1000\thesis1000\diagnosis\test-distspecto-txt'
os.makedirs(test_specto_txt_dir, exist_ok=True)
for i in range(3):
    with open(os.path.join(test_specto_txt_dir, f'dummy_{i}.txt'), 'w') as f:
        f.write("This is a dummy transcript for testing.\n")

print("Script finished")