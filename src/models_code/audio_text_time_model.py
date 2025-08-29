import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from utils import pad_sequences_and_times_np
from weights import Weights
from tensorflow.keras.models import load_model
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from config import PROCESSED_DATA_PATH, VOCAB_PATH, WORD2VEC_PATH, MAX_SEQUENCE_LENGTH
AUDIO_FEATURES_CACHE_PATH = "precomputed_audio_features.npy"
LABELS_CACHE_PATH = "precomputed_labels.npy"
FEATURE_EXTRACTION_BATCH_SIZE = 16

# --- MODEL DEFINITION ---
def create_combined_model(embedding_layer, max_sequence_length, audio_feature_shape):
    # Audio model branch
    audio_input = Input(shape=audio_feature_shape, dtype=tf.float32, name="audio_input")
    audio_output = Dropout(0.5)(audio_input)
    audio_model = Model(inputs=audio_input, outputs=audio_output, name="audio_model")
    audio_model.summary()
    # Word/Time model branch
    word_input = Input(shape=(max_sequence_length,), dtype=tf.int32, name="word_input")
    time_stamps = Input(shape=(max_sequence_length, 2), dtype=tf.float32, name="time_stamps")
    word_embedded = embedding_layer(word_input)
    concatenated = Concatenate()([word_embedded, time_stamps])
    lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)
    word_time_model = Model(inputs=[word_input, time_stamps], outputs=lstm_output, name="word_model")

    # Concatenate the outputs
    combined_output = Concatenate()([audio_model.output, word_time_model.output])
    final_output = Dense(1, activation='sigmoid')(combined_output)

    # Build the combined model
    combined_model = Model(
        inputs=[audio_model.input, word_time_model.input[0], word_time_model.input[1]],
        outputs=final_output,
        name="combined_model"
    )
    return combined_model

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run split_data.py first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    if os.path.exists(AUDIO_FEATURES_CACHE_PATH) and os.path.exists(LABELS_CACHE_PATH):
        print("Loading pre-computed audio features from cache...")
        all_audios = np.load(AUDIO_FEATURES_CACHE_PATH)
        all_labels = np.load(LABELS_CACHE_PATH)
    else:
        raw_audios = np.stack([d['audio'] for d in data_points])
        all_labels = np.array([1 if d['label'] == 'dementia' else 0 for d in data_points])

        # Instantiate the feature extractor
        model_checkpoint = "facebook/wav2vec2-base"
        feature_extractor = Wav2VecFeatureExtractor(model_checkpoint)
        pooling_layer = GlobalAveragePooling1D()

        print(f"Processing audio files in batches of {FEATURE_EXTRACTION_BATCH_SIZE}...")
        extracted_features_list = []
        for i in range(0, len(raw_audios), FEATURE_EXTRACTION_BATCH_SIZE):
            batch_raw_audios = raw_audios[i:i + FEATURE_EXTRACTION_BATCH_SIZE]
            batch_features = feature_extractor(batch_raw_audios)
            batch_features_pooled = pooling_layer(batch_features)
            extracted_features_list.append(batch_features_pooled)
            print(f"Processed batch {i // FEATURE_EXTRACTION_BATCH_SIZE + 1} of {len(raw_audios) // FEATURE_EXTRACTION_BATCH_SIZE + 1}...")

        all_audios = tf.concat(extracted_features_list, axis=0).numpy()

        print("Feature extraction complete. Caching features for future runs...")
        np.save(AUDIO_FEATURES_CACHE_PATH, all_audios)
        np.save(LABELS_CACHE_PATH, all_labels)
        print("Features cached successfully.")
    
    # Extract word and time data
    all_words = [d['words'] for d in data_points]
    all_times = [d['word_times'] for d in data_points]

    if not os.path.exists(VOCAB_PATH) or not os.path.exists(WORD2VEC_PATH):
        raise FileNotFoundError("Vocab or Word2Vec vectors not found. Please run build_vocab.py first.")
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
    
    audio_feature_shape = all_audios.shape[1:]

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_results = []
    fold_number = 1

    # Loop through each of the 5 folds
    for train_val_index, test_index in kf.split(all_word_ids):
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        # Create train/val/test splits for this fold
        audio_train_val = all_audios[train_val_index]
        word_train_val = [all_word_ids[i] for i in train_val_index]
        time_train_val = [all_times[i] for i in train_val_index]
        y_train_val = all_labels[train_val_index]

        audio_test = all_audios[test_index]
        word_test = [all_word_ids[i] for i in test_index]
        time_test = [all_times[i] for i in test_index]
        y_test = all_labels[test_index]

        train_size = int(len(word_train_val) * 0.8)
        audio_train = audio_train_val[:train_size]
        word_train = word_train_val[:train_size]
        time_train = time_train_val[:train_size]
        y_train = y_train_val[:train_size]
        audio_val = audio_train_val[train_size:]
        word_val = word_train_val[train_size:]
        time_val = time_train_val[train_size:]
        y_val = y_train_val[train_size:]
        
        # Pad the sequences
        word_train_padded, time_train_padded = pad_sequences_and_times_np(word_train, time_train, MAX_SEQUENCE_LENGTH)
        word_val_padded, time_val_padded = pad_sequences_and_times_np(word_val, time_val, MAX_SEQUENCE_LENGTH)
        word_test_padded, time_test_padded = pad_sequences_and_times_np(word_test, time_test, MAX_SEQUENCE_LENGTH)
        
        # Standardize time features
        time_mean = time_train_padded.mean(axis=(0,1))
        time_std = time_train_padded.std(axis=(0,1))
        time_train_padded = (time_train_padded - time_mean) / (time_std + 1e-8)
        time_val_padded = (time_val_padded - time_mean) / (time_std + 1e-8)
        time_test_padded = (time_test_padded - time_mean) / (time_std + 1e-8)
        
        # Re-initialize and compile a new model for each fold
        model = create_combined_model(embedding_layer, MAX_SEQUENCE_LENGTH, audio_feature_shape)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        # Calculate class weights
        y_train_flat = y_train.flatten()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        # decreased a bit the patience for faster computation
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Create a directory to save the models
        model_save_dir = "models"
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Define the ModelCheckpoint callback to save the best model for this fold
        checkpoint_path = os.path.join(model_save_dir, f"audio_text_time_model_fold_{fold_number}.keras")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        model.fit(x=[audio_train, word_train_padded, time_train_padded], y=y_train,
                  validation_data=([audio_val, word_val_padded, time_val_padded], y_val),
                  epochs=25,
                  batch_size=16,
                  shuffle=True,
                  callbacks=[callback, model_checkpoint_callback],
                  class_weight=class_weight_dict,
                  verbose=0)
        
        model.summary()

        # Evaluate on the test data
        eval_results = model.evaluate([audio_test, word_test_padded, time_test_padded], y_test)
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
    eval_results_array = np.array(all_eval_results)

    # The accuracy metric is at index 1 in the evaluation results list
    accuracy_results = eval_results_array[:, 1]

    # Find the index of the fold with the highest accuracy
    best_accuracy_index = np.argmax(accuracy_results)

    # The fold number is the index plus 1
    best_fold_number = best_accuracy_index + 1

    # Print the results for clarity
    print("\n--- Identifying the Best Model ---")
    print(f"✅ The best model was found in Fold {best_fold_number} with an accuracy of {accuracy_results[best_accuracy_index]:.4f}")

    # Construct the file path for the best model
    best_model_path = os.path.join(model_save_dir, f"audio_text_time_model_fold_{best_fold_number}.keras")

    # Load the best-performing model
    try:
        best_model_for_prediction = load_model(best_model_path)
        print("✅ Successfully loaded the best model.")
        
        # Save the best model to a new, more descriptive file name
        final_save_path = os.path.join(model_save_dir, "best_model_audio_text_time.keras")
        best_model_for_prediction.save(final_save_path)
        print(f"✅ The best model has been saved to: {final_save_path}")

    except Exception as e:
        print(f"❌ Error loading or saving the model: {e}")