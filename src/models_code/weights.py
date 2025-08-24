import numpy as np

class Weights:
    def __init__(self, vocab, word2vec_vectors):
        self.vocab = vocab
        self.word2vec_vectors = word2vec_vectors
    def get_weight_matrix(self):
        embedding_dim = self.word2vec_vectors.vector_size
        weight_matrix = np.zeros((len(self.vocab) + 1, embedding_dim))
        
        # Iterate through the words in your vocabulary to find their corresponding vectors.
        for word, index in self.vocab.items():
            if word in self.word2vec_vectors:
                # Assign the pre-trained vector to the corresponding row in the matrix.
                weight_matrix[index] = self.word2vec_vectors[word]
            # No 'else' needed; words not found will retain their initial zero vector.
                
        return weight_matrix
