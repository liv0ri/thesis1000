from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, Input

from keras.models import Sequential
from gensim.models import KeyedVectors
word2vec_vectors = KeyedVectors.load("/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/word2vec_embeddings/word2vec.wordvectors")

vocab = tokenizer.word_index

embedding_vectors = get_weight_matrix()

embedding_layer = Embedding(input_dim=len(vocab) + 1,
                                output_dim=300,
                                weights=[embedding_vectors],
                                input_length=50,
                                trainable=False)

from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Embedding, Concatenate, Reshape, LSTM, Dense
# Define input layers
word_input = Input(shape=(50))

# Define embedding layer

# Embed word and pos inputs
word_embedded = embedding_layer(word_input)

# Concatenate word and pos embeddings

# Reshape the concatenated tensor
#reshaped = Reshape((-1, 16))(concatenated)

# Apply LSTM layer
lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(word_embedded)

# Apply dense layer for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

#opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Define the model
model0 = Model(inputs=word_input, outputs=output)

# Compile the model
model0.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# Print the model summary
model0.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10)
model0.fit(word_train, y_train, shuffle =True, validation_data=(word_val, y_val), epochs=50, batch_size=16, callbacks=[callback])

model0.evaluate(word_test, y_test)

## Fit the model
model0.save("/content/drive/MyDrive/Colab_Notebooks/dementia/English/Pitt-xml/dementia_lstm_model.h5")

import tensorflow as tf
model = tf.keras.models.load_model('/content/drive/MyDrive/Colab_Notebooks/dementia/English/dementia/English/Pitt/trained_models_shorts_augmented/word')