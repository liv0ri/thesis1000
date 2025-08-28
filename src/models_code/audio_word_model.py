import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D, Concatenate, Embedding
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Import helper functions and classes from your other files
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from utils import pad_sequences_and_times_np
from weights import Weights

# --- CONFIG ---
PROCESSED_DATA_PATH = "processed_data.pkl"
VOCAB_PATH = "vocab.pkl"
WORD2VEC_PATH = "word2vec_vectors.pkl"
MAX_SEQUENCE_LENGTH = 50

def create_audio_word_model(embedding_layer, max_sequence_length):
    """
    Defines and returns the multimodal audio-word Keras model.
    It combines a Wav2Vec branch for audio features and an LSTM branch for word embeddings.
    """
    model_checkpoint = "facebook/wav2vec2-base"
    audio_input = Input(shape=(16000,), name="audio_input")

    # Audio model branch
    # The Wav2VecFeatureExtractor is a Keras Layer that processes raw audio
    # and extracts high-level features, which are then pooled.
    wav2vec_extractor = Wav2VecFeatureExtractor(model_checkpoint)
    audio_features = wav2vec_extractor(audio_input)
    audio_pooled = GlobalAveragePooling1D()(audio_features)
    audio_pooled = Dropout(0.5)(audio_pooled)
    audio_model = Model(inputs=audio_input, outputs=audio_pooled, name='audio_model')
    audio_model.summary()
    
    # Word model branch
    # This branch uses a pre-trained word embedding layer and an LSTM to process text.
    word_input = Input(shape=(max_sequence_length,), dtype=tf.int32, name="word_input")
    word_embedded = embedding_layer(word_input)
    word_lstm = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(word_embedded)
    word_model = Model(inputs=word_input, outputs=word_lstm, name='word_model')

    # Concatenate the outputs of the two branches to create the multimodal model
    combined = Concatenate()([audio_model.output, word_model.output])
    combined_output = Dense(1, activation='sigmoid')(combined)
    
    # Build the combined model with both audio and word inputs
    audio_word_model = Model(inputs=[audio_model.input, word_model.input], outputs=combined_output, name='audio_word_model')
    return audio_word_model

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run the data preparation script first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    # Extract all data points from the loaded file
    all_audios = np.array([d['audio'] for d in data_points])
    all_words = [d['words'] for d in data_points]
    all_labels = np.array([1 if d['label'] == 'dementia' else 0 for d in data_points])

    if not os.path.exists(VOCAB_PATH) or not os.path.exists(WORD2VEC_PATH):
        raise FileNotFoundError("Vocab or Word2Vec vectors not found. Please run the build_vocab script first.")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(WORD2VEC_PATH, "rb") as f:
        word2vec_vectors = pickle.load(f)
    
    word_to_id = {word: i for i, word in enumerate(vocab)}
    all_word_ids = [[word_to_id.get(w, 0) for w in sentence] for sentence in all_words]

    weight = Weights(vocab, word2vec_vectors)
    embedding_vectors = weight.get_weight_matrix()
    embedding_dim = embedding_vectors.shape[1]
    
    embedding_layer = Embedding(input_dim=len(vocab) + 1,
                                output_dim=embedding_dim,
                                weights=[embedding_vectors],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_results = []
    fold_number = 1
    
    # Create a directory to save the models for each fold
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Loop through each of the 5 folds
    for train_val_index, test_index in kf.split(all_audios):
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        # Create train/val/test splits for this fold
        audio_train_val = all_audios[train_val_index]
        word_train_val = [all_word_ids[i] for i in train_val_index]
        y_train_val = all_labels[train_val_index]

        audio_test = all_audios[test_index]
        word_test = [all_word_ids[i] for i in test_index]
        y_test = all_labels[test_index]

        train_size = int(len(audio_train_val) * 0.8)
        audio_train = audio_train_val[:train_size]
        word_train = word_train_val[:train_size]
        y_train = y_train_val[:train_size]
        audio_val = audio_train_val[train_size:]
        word_val = word_train_val[train_size:]
        y_val = y_train_val[train_size:]
        
        word_train_padded, _ = pad_sequences_and_times_np(word_train, None, MAX_SEQUENCE_LENGTH)
        word_val_padded, _ = pad_sequences_and_times_np(word_val, None, MAX_SEQUENCE_LENGTH)
        word_test_padded, _ = pad_sequences_and_times_np(word_test, None, MAX_SEQUENCE_LENGTH)
        
        # Re-initialize and compile a new model for each fold
        model = create_audio_word_model(embedding_layer, MAX_SEQUENCE_LENGTH)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        # Calculate class weights
        y_train_flat = y_train.flatten()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Define the ModelCheckpoint callback for this fold
        checkpoint_path = os.path.join(model_save_dir, f"audio_word_model_fold_{fold_number}.keras")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        model.fit([audio_train, word_train_padded], y_train,
                  validation_data=([audio_val, word_val_padded], y_val),
                  epochs=25,
                  batch_size=16,
                  shuffle=True,
                  callbacks=[callback, model_checkpoint_callback],
                  class_weight=class_weight_dict)
        eval_results = model.evaluate([audio_test, word_test_padded], y_test)
        all_eval_results.append(eval_results)
        
        print(f"Fold {fold_number} Test Results: {eval_results}")
        fold_number += 1

    # Calculate and print the average results across all folds
    avg_results = np.mean(all_eval_results, axis=0)
    std_results = np.std(all_eval_results, axis=0)

    print("\n--- Final Results (Mean ± Std Dev) ---")
    print(f"Loss: {avg_results[0]:.4f} ± {std_results[0]:.4f}")
    print(f"Accuracy: {avg_results[1]:.4f} ± {std_results[1]:.4f}")
    print(f"Precision: {avg_results[2]:.4f} ± {std_results[2]:.4f}")
    print(f"Recall: {avg_results[3]:.4f} ± {std_results[3]:.4f}")
    print(f"AUC: {avg_results[4]:.4f} ± {std_results[4]:.4f}")
    
    print("\n✅ Finished all folds.")

    # Find and save the single best model overall
    eval_results_array = np.array(all_eval_results)
    best_accuracy_index = np.argmax(eval_results_array[:, 1])
    best_fold_number = best_accuracy_index + 1
    
    print("\n--- Identifying the Best Model ---")
    print(f"✅ The overall best model was found in Fold {best_fold_number}.")
    
    # Load the best-performing model from its saved location
    best_model_path = os.path.join(model_save_dir, f"audio_word_model_fold_{best_fold_number}.keras")
    best_model_for_prediction = load_model(best_model_path)
    
    # Save it to a new, more descriptive filename for final use
    final_save_path = os.path.join(model_save_dir, "best_audio_word_model_overall.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"✅ The best model has been saved to: {final_save_path}")