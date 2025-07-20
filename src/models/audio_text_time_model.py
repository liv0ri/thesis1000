import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from transformers import TFAutoModel
from wav2vec_feature_extractor import Wav2VecFeatureExtractor

vocab = {f"word{i}": i for i in range(1, 101)}

audio_train = np.random.randn(10, 16000).astype(np.float32)
word_train = np.random.randint(1, len(vocab), size=(10, 50))
time_train = np.random.randn(10, 50, 2).astype(np.float32)
y_train = np.random.randint(0, 2, size=(10, 1))

audio_val = np.random.randn(2, 16000).astype(np.float32)
word_val = np.random.randint(1, len(vocab), size=(2, 50))
time_val = np.random.randn(2, 50, 2).astype(np.float32)
y_val = np.random.randint(0, 2, size=(2, 1))

audio_test = np.random.randn(2, 16000).astype(np.float32)
word_test = np.random.randint(1, len(vocab), size=(2, 50))
time_test = np.random.randn(2, 50, 2).astype(np.float32)
y_test = np.random.randint(0, 2, size=(2, 1))

combined_train = [word_train, time_train]
combined_val = [word_val, time_val]
combined_test = [word_test, time_test]

# Load the pre-trained model using the Hugging Face interface
model_checkpoint = "facebook/wav2vec2-base"
# Define the inputs to the model
audio_input = Input(shape=(16000,), dtype=tf.float32)
audio_features = Wav2VecFeatureExtractor(model_checkpoint)(audio_input)
audio_output = GlobalAveragePooling1D()(audio_features)
audio_output = Dropout(0.5)(audio_output)

# Create the TensorFlow functional API model
audio_model = Model(inputs=audio_input, outputs=audio_output, name="audio_model")

# Print the model summary
audio_model.summary()

word_input = Input(shape=(50,), dtype=tf.int32)
time_stamps = Input(shape=(50, 2), dtype=tf.float32)


# word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors", mmap='r')

# # Save the vocabulary dictionary
# with open('/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original_augmented/vocab_dict.pkl', 'wb') as fp:
#     pickle.dump(vocab, fp)
#     print('dictionary saved successfully to file')

# # Import the vocabulary dictionary
# with open('/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original_augmented/vocab_dict.pkl', 'rb') as handle:
#     data = handle.read()
# vocab = pickle.loads(data)

# vocab = tokenizer.word_index

# # Get the word embeddings for each word
# #vocab = tokenizer.word_index
class DummyTokenizer:
    def __init__(self, vocab):
        self.word_index = vocab

tokenizer = DummyTokenizer(vocab)

word2vec_vectors = {key: np.random.rand(300) for key in vocab}

def get_weight_matrix():
    weight_matrix = np.zeros((len(vocab)+1, 300))
    for key, idx in vocab.items():
        weight_matrix[idx] = word2vec_vectors.get(key, np.zeros(300))
    return weight_matrix

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

embedding_vectors = get_weight_matrix()

embedding_layer = Embedding(input_dim=len(vocab) + 1,
                             output_dim=300,
                             weights=[embedding_vectors],
                             input_length=50,
                             trainable=False)

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])
# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)
# Define the model
word_model = Model(inputs=[word_input, time_stamps], outputs=lstm_output, name="word_model")

# Print the model summary
word_model.summary()
combined_output = Concatenate()([audio_model.output, word_model.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
final_output = Dense(1, activation='sigmoid')(combined_output)
combined_model = Model(inputs=[audio_model.input, word_model.input], outputs=final_output, name="combined_model")
combined_model.summary()

# Compile and train the model
combined_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
combined_model.fit([audio_train, combined_train], y_train,
                   epochs=50, batch_size=16, validation_data=([audio_val, combined_val], y_val),
                  callbacks=[callback])

# Evaluate the model using the test set
combined_model.evaluate([audio_test, combined_test], y_test)

os.makedirs("models", exist_ok=True)
combined_model.save("models/audio_word_time.keras")
