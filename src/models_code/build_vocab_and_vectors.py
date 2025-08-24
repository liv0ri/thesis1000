import os
import pickle
import collections
from gensim.models import Word2Vec
import numpy as np

# --- User-defined variables ---
DATA_BASE_PATH = "pitt_split1" 
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
    if not os.path.exists(folder_path):
        print(f"Error: Transcript folder not found at {folder_path}")
        return sentences

    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".pkl"):
                file_path = os.path.join(root, fname)
                try:
                    with open(file_path, "rb") as f:
                        transcript = pickle.load(f)
                    
                    # Filter out None values from the transcript lists
                    cleaned_transcript = [[word for word in sentence if word is not None] for sentence in transcript]
                    # Filter out any empty sentences that might result from the cleaning
                    cleaned_transcript = [sentence for sentence in cleaned_transcript if sentence]
                    
                    sentences.extend(cleaned_transcript)
                except (IOError, pickle.UnpicklingError) as e:
                    print(f"Error reading {file_path}: {e}")
    return sentences

def build_vocabulary(sentences):
    """
    Builds a vocabulary of all unique words from a list of sentences
    and maps each word to an integer index, including an OOV token.
    
    Args:
        sentences (list): A list of sentences (lists of words).
        
    Returns:
        dict: A dictionary mapping each unique word to a unique integer index.
    """
    word_counts = collections.Counter()
    for sentence in sentences:
        # Filter out None values just in case
        clean_sentence = [word for word in sentence if word is not None]
        word_counts.update(clean_sentence)
    
    # Get a list of all unique words from the word counts
    unique_words = sorted(word_counts.keys())
    vocab_dict = {}
    
    # Create the dictionary by assigning a unique index to each word
    for i, word in enumerate(unique_words):
        vocab_dict[word] = i + 1
    
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
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
        epochs=100
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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Successfully saved data to {file_path}")

if __name__ == "__main__":
    TRANSCRIPTS_FOLDER = os.path.join(DATA_BASE_PATH, "transcripts")
    VOCAB_OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, "vocab.pkl")
    WORD2VEC_OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, "word2vec_vectors.pkl")
    
    # Step 1: Load all sentences from the transcript files.
    sentences = get_transcripts_from_folder(TRANSCRIPTS_FOLDER)
    
    if not sentences:
        print("No transcripts found. Cannot proceed.")
    else:
        # Step 2: Build the full vocabulary, including the '<unk>' token.
        vocab = build_vocabulary(sentences)
        
        # Step 3: Create a new list of sentences where words not in the vocabulary
        # are replaced with the '<unk>' token.
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentence = [word for word in sentence if word in vocab]
            tokenized_sentences.append(tokenized_sentence)
        
        # Step 4: Train the Word2Vec model on the tokenized sentences.
        # This ensures the model learns a vector for the '<unk>' token.
        word2vec_model = train_word2vec_model(tokenized_sentences)
        
        # Step 5: Save both the vocabulary and the word2vec vectors.
        save_data(vocab, VOCAB_OUTPUT_PATH)
        word2vec_vectors = word2vec_model.wv
        save_data(word2vec_vectors, WORD2VEC_OUTPUT_PATH)
