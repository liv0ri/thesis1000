import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import pad_sequences_and_times_np
from weights import Weights
from config import PROCESSED_DATA_PATH, VOCAB_PATH, WORD2VEC_PATH, MAX_SEQUENCE_LENGTH

def create_model(embedding_layer, max_sequence_length):
    word_input = Input(shape=(max_sequence_length,), name='word_input', dtype=tf.int32)
    time_stamps = Input(shape=(max_sequence_length, 2), name='time_input', dtype=tf.float32)
    word_embedded = embedding_layer(word_input)
    concatenated = Concatenate()([word_embedded, time_stamps])
    lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(concatenated)
    output = Dense(1, activation='sigmoid')(lstm_output)
    model = Model(inputs=[word_input, time_stamps], outputs=output)
    return model

def train_and_save_text_time_model(dataset_type, remove_short_sentences):
    cache_base_name = f"precomputed_{dataset_type}"
    if remove_short_sentences:
        cache_base_name += "_no_short"
    LABELS_CACHE_PATH = f"{cache_base_name}_labels.npy"

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run the data preparation script first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    # Filter data based on input parameters
    if dataset_type == "original":
        data_points = [d for d in data_points if d['source_type'] == 'original']
    elif dataset_type == "augmented":
        data_points = [d for d in data_points if d['source_type'] == 'text_augmented']
    if remove_short_sentences:
        data_points = [d for d in data_points if len(d['words']) >= 5]

    if os.path.exists(LABELS_CACHE_PATH):
        all_labels = np.load(LABELS_CACHE_PATH)
    else:
        all_labels = np.array([1 if d['label'] == 'dementia' else 0 for d in data_points])
        np.save(LABELS_CACHE_PATH, all_labels)
    
    # Extract data for the model
    all_words = [d['words'] for d in data_points]
    # Correctly extract the word_times field, which contains timestamps for each word.
    all_times = [d['word_times'] for d in data_points]
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
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_results = []
    fold_number = 1
    
    model_save_dir = f"models\{dataset_type}"
    if remove_short_sentences:
        model_save_dir += "_no_short"
    os.makedirs(model_save_dir, exist_ok=True)

    # Loop through each of the 5 folds
    for train_val_index, test_index in kf.split(all_word_ids, all_labels):
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        # Create train/val/test splits for this fold
        word_train_val = [all_word_ids[i] for i in train_val_index]
        time_train_val = [all_times[i] for i in train_val_index]
        y_train_val = all_labels[train_val_index]
        
        word_test = [all_word_ids[i] for i in test_index]
        time_test = [all_times[i] for i in test_index]
        y_test = all_labels[test_index]
        
        word_train, word_val, time_train, time_val, y_train, y_val = train_test_split(
            word_train_val,
            time_train_val,
            y_train_val,
            test_size=0.2,
            stratify=y_train_val,
            random_state=42
        )
        
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
        model = create_model(embedding_layer, MAX_SEQUENCE_LENGTH)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        # Calculate class weights
        y_train_flat = y_train.flatten()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Early stopping callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        checkpoint_path = os.path.join(model_save_dir, f"text_time_model_fold_{fold_number}.keras")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )

        # Train the model
        model.fit([word_train_padded, time_train_padded], y_train,
                  validation_data=([word_val_padded, time_val_padded], y_val),
                  epochs=25,
                  batch_size=16,
                  shuffle=True,
                  callbacks=[callback, model_checkpoint_callback],
                  class_weight=class_weight_dict,
                  verbose=0)
        
        eval_results = model.evaluate([word_test_padded, time_test_padded], y_test)
        all_eval_results.append(eval_results)
        
        print(f"Fold {fold_number} Test Results: {eval_results}")
        fold_number += 1

    # Calculate and print the average results across all folds
    avg_results = np.mean(all_eval_results, axis=0)
    std_results = np.std(all_eval_results, axis=0)

    print(f"Loss: {avg_results[0]:.4f} ± {std_results[0]:.4f}")
    print(f"Accuracy: {avg_results[1]:.4f} ± {std_results[1]:.4f}")
    print(f"Precision: {avg_results[2]:.4f} ± {std_results[2]:.4f}")
    print(f"Recall: {avg_results[3]:.4f} ± {std_results[3]:.4f}")
    print(f"AUC: {avg_results[4]:.4f} ± {std_results[4]:.4f}")

    eval_results_array = np.array(all_eval_results)
    
    # With AUC-based selection:
    auc_results = eval_results_array[:, 4]  # AUC is at index 4 in your metrics
    best_auc_index = np.argmax(auc_results)
    best_fold_number = best_auc_index + 1

    print(f"The best model was found in Fold {best_fold_number} with an AUC of {auc_results[best_auc_index]:.4f}")
    
    # Load the best-performing model from its saved location
    best_model_path = os.path.join(model_save_dir, f"text_time_model_fold_{best_fold_number}.keras")
    best_model_for_prediction = load_model(best_model_path)
    
    # Save it to a new, more descriptive filename for final use
    final_save_path = os.path.join(model_save_dir, "best_text_time_model.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"The best model has been saved to: {final_save_path}")

if __name__ == "__main__":
    train_and_save_text_time_model(dataset_type="original", remove_short_sentences=False)
    train_and_save_text_time_model(dataset_type="both", remove_short_sentences=False)
