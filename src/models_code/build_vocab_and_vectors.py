import os
import pickle
import collections
from gensim.models import Word2Vec
import numpy as np

# --- User-defined variables ---
# Set the ABSOLUTE path where your processed data (from the cha-file-processor) is located.
# Example: C:\\Users\\YourName\\Documents\\my_project\\processed_data
DATA_BASE_PATH = "pitt_split1" 
# Set the ABSOLUTE path where you want the output files (vocab and word2vec) to be saved.
# Example: C:\\Users\\YourName\\Documents\\my_project\\embeddings
OUTPUT_BASE_PATH = "pitt_split"

def get_transcripts_from_folder(folder_path):
    """
    Recursively reads all .pkl transcript files from the specified folder
    and its subdirectories, extracts the text content, and returns a 
    list of all sentences.
    
    Args:
        folder_path (str): The path to the directory containing transcript files.
        
    Returns:
        list: A list of sentences, where each sentence is a list of words.
    """
    sentences = []
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Transcript folder not found at {folder_path}")
        return sentences

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".pkl"):
                file_path = os.path.join(root, fname)
                try:
                    # Transcripts are stored as lists of lists of words.
                    with open(file_path, "rb") as f:
                        transcript = pickle.load(f)
                    
                    # Extend the main list with the transcripts from this file
                    sentences.extend(transcript)
                except (IOError, pickle.UnpicklingError) as e:
                    print(f"Error reading {file_path}: {e}")
    return sentences

def build_vocabulary(sentences):
    """
    Builds a vocabulary of all unique words from a list of sentences
    and maps each word to an integer index.
    
    Args:
        sentences (list): A list of sentences (lists of words).
        
    Returns:
        dict: A dictionary mapping each unique word to a unique integer index.
    """
    word_counts = collections.Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    # Create a dictionary mapping each word to a unique integer index
    vocab_dict = {word: i for i, word in enumerate(sorted(word_counts.keys()))}
    
    return vocab_dict

def train_word2vec_model(sentences):
    """
    Trains a Word2Vec model on the provided sentences.
    
    Args:
        sentences (list): A list of sentences (lists of words).
        
    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    print("Training Word2Vec model...")
    # Using sensible default parameters for a small dataset
    model = Word2Vec(
        sentences,
        vector_size=100,  # Dimensionality of the word vectors
        window=5,         # Maximum distance between the current and predicted word
        min_count=1,      # Ignores all words with total frequency lower than this
        sg=0,             # Training algorithm: 0 for CBOW, 1 for skip-gram
        epochs=100        # Number of iterations over the corpus
    )
    print("Training complete.")
    return model

def save_data(data, file_path):
    """
    Saves data to a file using pickle.
    
    Args:
        data: The data to save.
        file_path (str): The path where the file will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Successfully saved data to {file_path}")

if __name__ == "__main__":
    # The full path to the transcript files
    TRANSCRIPTS_FOLDER = os.path.join(DATA_BASE_PATH, "transcripts")
    # The full path to save the vocabulary file
    VOCAB_OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, "vocab.pkl")
    # The full path to save the word vectors
    WORD2VEC_OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, "word2vec_vectors.pkl")
    
    # Step 1: Get all sentences (transcripts) from the folder and its subdirectories
    sentences = get_transcripts_from_folder(TRANSCRIPTS_FOLDER)
    
    if not sentences:
        print("No transcripts found. Cannot proceed.")
    else:
        # Step 2: Build the vocabulary
        vocab = build_vocabulary(sentences)
        save_data(vocab, VOCAB_OUTPUT_PATH)
        
        # Step 3: Train the Word2Vec model
        word2vec_model = train_word2vec_model(sentences)
        
        # Step 4: Extract the word vectors and save them
        word2vec_vectors = word2vec_model.wv
        save_data(word2vec_vectors, WORD2VEC_OUTPUT_PATH)
