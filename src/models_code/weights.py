import numpy as np
class Weights:
    def __init__(self, vocab, word2vec_vectors):
        self.vocab = vocab
        self.word2vec_vectors = word2vec_vectors

    def get_weight_matrix(self):
        weight_matrix = np.zeros((len(self.vocab)+1, 300))
        for key, idx in self.vocab.items():
            weight_matrix[idx] = self.word2vec_vectors.get(key, np.zeros(300))
        return weight_matrix
    
    def get_weight_matrix(self):
        """
        Builds the embedding weight matrix from the vocabulary and word2vec vectors.
        It handles words that are not in the pre-trained word2vec vocabulary.
        """
        # The size of the word2vec vectors is now determined dynamically from the loaded model.
        embedding_dim = self.word2vec_vectors.vector_size
        # Initialize the weight matrix with zeros. The +1 is for the OOV token.
        weight_matrix = np.zeros((len(self.vocab) + 1, embedding_dim))
        
        # Iterate through the words in your vocabulary to find their corresponding vectors.
        for word, index in self.vocab.items():
            # Look up the word vector in the loaded KeyedVectors object.
            # Use bracket notation `[]` instead of `.get()`.
            vector = self.word2vec_vectors[word]
            weight_matrix[index] = vector

    
    # def get_weight_matrix():
#     # define weight matrix dimensions with all 0
#     weight_matrix = np.zeros((len(vocab)+1, word2vec_vectors.vector_size))
#     i=0
#     for key in vocab.keys():
#       if key=='OOV':
#         continue
#       elif key not in word2vec_vectors:
#         i=i+1
#         continue
#       else:
#         weight_matrix[i + 1] = word2vec_vectors[key]
#         i=i+1
#     return weight_matrix