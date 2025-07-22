import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from weights import Weights

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

# Load the pre-trained model using the Hugging Face interface
model_checkpoint = "facebook/wav2vec2-base"
# Define the inputs to the model
audio_input = Input(shape=(16000,), dtype=tf.float32)

# from gensim.models import KeyedVectors
# word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors", mmap='r')

# huggingface_model = TFAutoModel.from_pretrained(model_checkpoint, trainable=False, from_pt=True)
# # Pass the inputs hrough the Wav2Vec model
# wav2vec_output = huggingface_model(input_values)
audio_features = Wav2VecFeatureExtractor(model_checkpoint)(audio_input)
audio_output = GlobalAveragePooling1D()(audio_features)
audio_output = Dropout(0.5)(audio_output)

# Create the TensorFlow functional API model
audio_model = Model(inputs=audio_input, outputs=audio_output, name="audio_model")

# Print the model summary
audio_model.summary()

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

weight = Weights(vocab, word2vec_vectors)

embedding_vectors = weight.get_weight_matrix()

embedding_layer = Embedding(input_dim=len(vocab) + 1,
                             output_dim=300,
                             weights=[embedding_vectors],
                             input_length=50,
                             trainable=False)

word_input = Input(shape=(50,), dtype=tf.int32)
time_stamps = Input(shape=(50, 2), dtype=tf.float32)

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings
concatenated = Concatenate()([word_embedded, time_stamps])
# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)
# Define the model
word_time_model = Model(inputs=[word_input, time_stamps], outputs=lstm_output, name="word_model")

# Print the model summary
word_time_model.summary()
# IN CASE WWE REMOVE THE ONES ABOVE
# # # Reshape word_model output to match the shape of audio_model output
# audio_model_output = layers.GlobalAveragePooling1D()(audio_model.output[1])
# # Drop-out layer before the final Classification-Head
# audio_model_output = layers.Dropout(0.5) (audio_model_output)

combined_output = Concatenate()([audio_model.output, word_time_model.output])

final_output = Dense(1, activation='sigmoid')(combined_output)
combined_model = Model(inputs=[audio_model.input, word_time_model.input], outputs=final_output, name="combined_model")
combined_model.summary()

combined_train = [word_train, time_train]
combined_val = [word_val, time_val]
combined_test = [word_test, time_test]

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

# Save the model to the models directory
audio_model.save(os.path.join("models", "audio_word_time_model.keras"))
