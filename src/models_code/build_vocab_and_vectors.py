import os
import pickle
import collections
from gensim.models import Word2Vec

OUTPUT_BASE_PATH = "pitt_split" 
DATA_BASE_PATH = "pitt_split"  # base folder containing train/val/test

# ---------------- HELPERS ----------------
def get_transcripts_from_split(split_folder):
    """
    Recursively load all transcript .pkl files from a given split folder (train/val/test),
    including label subfolders (control/dementia).
    Returns a list of sentences (lists of words).
    """
    sentences = []
    transcripts_folder = os.path.join(split_folder, "transcripts")
    if not os.path.exists(transcripts_folder):
        raise FileNotFoundError(f"Transcript folder not found at {transcripts_folder}")

    for root, _, files in os.walk(transcripts_folder):
        for fname in files:
            if fname.endswith(".pkl"):
                file_path = os.path.join(root, fname)
                try:
                    with open(file_path, "rb") as f:
                        transcript = pickle.load(f)
                    if transcript:  # skip empty transcripts
                        sentences.append(transcript)
                except (IOError, pickle.UnpicklingError) as e:
                    print(f"Warning: could not read {file_path}: {e}")
    return sentences

def get_all_sentences(data_base_path):
    """
    Load transcripts from train, val, and test folders (all labels).
    Returns a combined list of sentences.
    """
    all_sentences = []
    for split in ["train", "val", "test"]:
        split_folder = os.path.join(data_base_path, split)
        sentences = get_transcripts_from_split(split_folder)
        all_sentences.extend(sentences)
    return all_sentences

def build_vocabulary(sentences):
    """
    Build a vocabulary dictionary mapping words to unique indices.
    """
    word_counts = collections.Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    unique_words = sorted(word_counts.keys())
    vocab_dict = {word: i + 1 for i, word in enumerate(unique_words)}
    return vocab_dict

def train_word2vec_model(sentences):
    """
    Train a Word2Vec model on tokenized sentences.
    """
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,     # CBOW
        epochs=100
    )
    return model

def save_data(data, file_path):
    """
    Save data (pickle) to the specified file path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    VOCAB_OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, "vocab.pkl")
    WORD2VEC_OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, "word2vec_vectors.pkl")

    # Load all sentences from train, val, test (including all labels)
    sentences = get_all_sentences(DATA_BASE_PATH)
    
    if not sentences:
        raise ValueError("No transcripts found in train/val/test. Cannot proceed.")

    # Build vocabulary
    vocab = build_vocabulary(sentences)

    # Tokenize sentences (redundant here but keeps consistency)
    tokenized_sentences = [[word for word in sentence if word in vocab] for sentence in sentences]

    # Train Word2Vec model
    word2vec_model = train_word2vec_model(tokenized_sentences)

    # Save outputs
    save_data(vocab, VOCAB_OUTPUT_PATH)
    word2vec_vectors = word2vec_model.wv
    save_data(word2vec_vectors, WORD2VEC_OUTPUT_PATH)

    print("✅ Vocabulary and Word2Vec vectors saved successfully.")
