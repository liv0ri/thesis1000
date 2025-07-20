import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from gensim.models import KeyedVectors

# word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors")

# vocab = tokenizer.word_index

vocab_size = 1000
embedding_dim = 300

# Dummy vocab simulating word_index (words mapped to indices)
vocab = {f"word{i}": i for i in range(1, vocab_size + 1)}

# Simulate word2vec vectors: a dictionary mapping word -> random 300-dim vector
word2vec_vectors = {word: np.random.rand(embedding_dim) for word in vocab.keys()}

# Function to create embedding matrix
def get_weight_matrix():
    weight_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    for word, idx in vocab.items():
        if word in word2vec_vectors:
            weight_matrix[idx] = word2vec_vectors[word]
        else:
            weight_matrix[idx] = np.zeros(embedding_dim)
    return weight_matrix

embedding_vectors = get_weight_matrix()

# Create embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=300,
                            weights=[embedding_vectors],
                            input_length=50,
                            trainable=False)

# Define input layer
word_input = Input(shape=(50,), name='word_input')

# Embed input
word_embedded = embedding_layer(word_input)

# LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(word_embedded)

# Apply dense layer for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

# Build model
model = Model(inputs=word_input, outputs=output, name='text_lstm_model')

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model.summary()

num_samples_train = 20
num_samples_val = 5
num_samples_test = 5

# Random integers from 1 to vocab_size for each word index in sequence length 50
word_train = np.random.randint(1, vocab_size + 1, size=(num_samples_train, 50))
y_train = np.random.randint(0, 2, size=(num_samples_train, 1))

word_val = np.random.randint(1, vocab_size + 1, size=(num_samples_val, 50))
y_val = np.random.randint(0, 2, size=(num_samples_val, 1))

word_test = np.random.randint(1, vocab_size + 1, size=(num_samples_test, 50))
y_test = np.random.randint(0, 2, size=(num_samples_test, 1))

# Early stopping callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
model.fit(word_train, y_train,
          validation_data=(word_val, y_val),
          epochs=10,
          batch_size=4,
          shuffle=True,
          callbacks=[callback])

# Evaluate on test set
model.evaluate(word_test, y_test)

# Save locally in 'models' folder
os.makedirs("models", exist_ok=True)
# Save the model to the folder
model.save(os.path.join("models", "text_model.keras"))
