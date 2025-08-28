import os
import pickle
import collections
from gensim.models import Word2Vec
from config import PROCESSED_DATA_PATH, VOCAB_PATH, WORD2VEC_PATH

def get_all_sentences(data_path):
    """Load transcripts from the processed data file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run split_data.py first.")
    
    with open(data_path, "rb") as f:
        data_points = pickle.load(f)

    all_sentences = [d['words'] for d in data_points]
    return all_sentences

def build_vocabulary(sentences):
    word_counts = collections.Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    unique_words = sorted(word_counts.keys())
    vocab_dict = {word: i + 1 for i, word in enumerate(unique_words)}
    return vocab_dict

def train_word2vec_model(sentences):
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
        epochs=100
    )
    return model

def save_data(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    sentences = get_all_sentences(PROCESSED_DATA_PATH)
    
    if not sentences:
        raise ValueError("No sentences found. Cannot proceed.")

    vocab = build_vocabulary(sentences)
    tokenized_sentences = [[word for word in sentence if word in vocab] for sentence in sentences]
    word2vec_model = train_word2vec_model(tokenized_sentences)
    word2vec_vectors = word2vec_model.wv

    save_data(vocab, VOCAB_PATH)
    save_data(word2vec_vectors, WORD2VEC_PATH)