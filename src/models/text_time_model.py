import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense
from tensorflow.keras.models import Model
import os
from weights import Weights

# word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors", mmap='r')

# Dummy vocab keys (like your vocab dict keys)
vocab = {f"word{i}": i for i in range(1, 1000 + 1)}

# import pickle
# # save dictionary to person_data.pkl file
# with open('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/final_combined_data/vocab_dict.pkl', 'wb') as fp:
#     pickle.dump(vocab, fp)
#     print('dictionary saved successfully to file')
# Load the vocabulary dictionary
# import pickle
# with open('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original/vocab_dict.pkl', 'rb') as handle:
#     data = handle.read()
# vocab = pickle.loads(data)
# vocab = tokenizer.word_index

# Dummy word2vec vectors as random vectors
word2vec_vectors = {word: np.random.rand(300) for word in vocab.keys()}

weight = Weights(vocab, word2vec_vectors)
embedding_vectors = weight.get_weight_matrix()

# Create embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=300,
                            weights=[embedding_vectors],
                            input_length=50,
                            trainable=False)

# Define inputs
word_input = Input(shape=(50,), name='word_input')


# final_combined_dataset=load_from_disk('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original')
# final_combined_dataset=final_combined_dataset.train_test_split(test_size=0.2)
# final_trained_dataset=final_combined_dataset['train'].train_test_split(test_size=0.2)
# word_train = np.asarray(final_trained_dataset['train']['word']).astype(np.float32)
# time_train = np.asarray(final_trained_dataset['train']['time_stamps']).astype(np.float32)
# word_val = np.asarray(final_trained_dataset['test']['word']).astype(np.float32)
# time_val = np.asarray(final_trained_dataset['test']['time_stamps']).astype(np.float32)
# y_train=np.asarray(final_trained_dataset['train']['label']).astype(np.float32)
# y_val=np.asarray(final_trained_dataset['test']['label']).astype(np.float32)
# word_test=np.asarray(final_combined_dataset['test']['word']).astype(np.float32)
# time_test=np.asarray(final_combined_dataset['test']['time_stamps']).astype(np.float32)
# y_test=np.asarray(final_combined_dataset['test']['label']).astype(np.float32)

# # Define input layers
# word_input = Input(shape=(50))
time_stamps = Input(shape=(50, 2), name='time_input')

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])

# LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)

# Apply dense layer for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

# Define the model
model = Model(inputs=[word_input, time_stamps], outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model.summary()

num_train = 20
num_val = 5
num_test = 5

# Random word indices in [1, 1000]
word_train = np.random.randint(1, 1000 + 1, size=(num_train, 50))
time_train = np.random.randn(num_train, 50, 2).astype(np.float32)
y_train = np.random.randint(0, 2, size=(num_train, 1))

word_val = np.random.randint(1, 1000 + 1, size=(num_val, 50))
time_val = np.random.randn(num_val, 50, 2).astype(np.float32)
y_val = np.random.randint(0, 2, size=(num_val, 1))

word_test = np.random.randint(1, 1000 + 1, size=(num_test, 50))
time_test = np.random.randn(num_test, 50, 2).astype(np.float32)
y_test = np.random.randint(0, 2, size=(num_test, 1))

# Train with EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit([word_train, time_train], y_train,
          validation_data=([word_val, time_val], y_val),
          epochs=50,
          batch_size=16,
          shuffle=True,
          callbacks=[callback])

# Evaluate on test data
model.evaluate([word_test, time_test], y_test)

# Save model locally
os.makedirs("models", exist_ok=True)
# Save the model to the folder
model.save(os.path.join("models", "word_time_model.keras"))