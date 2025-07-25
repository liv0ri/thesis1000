import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from transformers import TFAutoModel
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
import os
from weights import Weights

vocab = {f"word{i}": i for i in range(1, 101)}  # dummy vocab

audio_train = np.random.randn(10, 16000).astype(np.float32)
word_train = np.random.randint(1, len(vocab) + 1, size=(10, 50))
y_train = np.random.randint(0, 2, size=(10, 1))

audio_val = np.random.randn(2, 16000).astype(np.float32)
word_val = np.random.randint(1, len(vocab) + 1, size=(2, 50))
y_val = np.random.randint(0, 2, size=(2, 1))

audio_test = np.random.randn(2, 16000).astype(np.float32)
word_test = np.random.randint(1, len(vocab) + 1, size=(2, 50))
y_test = np.random.randint(0, 2, size=(2, 1))

word2vec_vectors = {key: np.random.rand(300) for key in vocab}
# word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors", mmap='r')
# # Load the vocabulary dictionary
# with open('/content/drive/MyDrive/Colab Notebooks/dementia/English/dementia/English/Pitt/final_combined_data_original_augmented/vocab_dict.pkl', 'rb') as handle:
#     data = handle.read()
# vocab = pickle.loads(data)
# #vocab = tokenizer.word_index
model_checkpoint = "facebook/wav2vec2-base"
# huggingface_model = TFAutoModel.from_pretrained(model_checkpoint, trainable=False, from_pt=True)

# Define the inputs to the model
input_values = Input(shape=(16000,), dtype=tf.float32)
audio_features = Wav2VecFeatureExtractor(model_checkpoint)(input_values)

# Create the TensorFlow functional API model
audio_model = Model(inputs=input_values, outputs=audio_features)

# Print the model summary
audio_model.summary()

weight = Weights(vocab, word2vec_vectors)
embedding_vectors = weight.get_weight_matrix()

# Create the embedding layer
embedding_layer = Embedding(input_dim=len(vocab) + 1,
                            output_dim=300,
                            weights=[embedding_vectors],
                            input_length=50,
                            trainable=False)

# Define input layers
word_input = Input(shape=(50,))

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)
# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(word_embedded)

# Define the model
word_model = Model(inputs=word_input, outputs=lstm_output, name='word_model')

# Print the model summary
word_model.summary()

# Reshape word_model output to match the shape of audio_model output
audio_model_output = GlobalAveragePooling1D()(audio_model.output)
# Drop-out layer before the final Classification-Head
audio_model_output = Dropout(0.5) (audio_model_output)

concatenated_output = Concatenate()([audio_model_output, word_model.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
audio_word= Dense(1, activation='sigmoid')(concatenated_output)
audio_word_model = Model(inputs=[audio_model.input, word_model.input], outputs=audio_word, name='combined_model')
audio_word_model.summary()

# Compile and train the model
audio_word_model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
audio_word_model.fit([audio_train, word_train], y_train,
                       epochs=50, batch_size=16, validation_data=([audio_val, word_val], y_val),
                       callbacks=[callback])

# Evaluate the model
audio_word_model.evaluate([audio_test, word_test], y_test)

audio_model.save(os.path.join("models", "audio_word_model.keras"))