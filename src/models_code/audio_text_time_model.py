import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import pad_sequences_and_times_np, prepare_audio_data
from weights import Weights
from config import PROCESSED_DATA_PATH, VOCAB_PATH, WORD2VEC_PATH, MAX_SEQUENCE_LENGTH


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


def train_and_save_multimodal_model(dataset_type, remove_short_sentences):
    cache_base_name = f"precomputed_{dataset_type}"
    if remove_short_sentences:
        cache_base_name += "_no_short"
    AUDIO_FEATURES_CACHE_PATH = f"{cache_base_name}_audio_features.npy"
    LABELS_CACHE_PATH = f"{cache_base_name}_labels.npy"

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run split_data.py first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    # Filter by dataset type and remove short sentences
    if dataset_type == "original":
        data_points = [d for d in data_points if d['source_type'] == 'original']
    elif dataset_type == "augmented":
        data_points = [d for d in data_points if d['source_type'] == 'text_augmented']
    if remove_short_sentences:
        data_points = [d for d in data_points if len(d['words']) >= 5]

    if os.path.exists(AUDIO_FEATURES_CACHE_PATH) and os.path.exists(LABELS_CACHE_PATH):
        all_audios = np.load(AUDIO_FEATURES_CACHE_PATH)
        all_labels = np.load(LABELS_CACHE_PATH)
    else:
        all_audios, all_labels = prepare_audio_data(data_points, AUDIO_FEATURES_CACHE_PATH, LABELS_CACHE_PATH)
    
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
    all_word_ids = [[word_to_id.get(w, 0) for w in sent] for sent in all_words]

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
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_number = 1
    all_eval_results = []

    model_save_dir = f"models\{dataset_type}"
    if remove_short_sentences:
        model_save_dir += "_no_short"
    os.makedirs(model_save_dir, exist_ok=True)

    for train_val_index, test_index in kf.split(all_word_ids, all_labels):
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

        audio_train, audio_val, word_train, word_val, time_train, time_val, y_train, y_val = train_test_split(
            audio_train_val,
            word_train_val,
            time_train_val,
            y_train_val,
            test_size=0.2,
            stratify=y_train_val,
            random_state=42
        )

        audio_mean = audio_train.mean(axis=0)
        audio_std = audio_train.std(axis=0)
        audio_train = (audio_train - audio_mean) / (audio_std + 1e-8)
        audio_val = (audio_val - audio_mean) / (audio_std + 1e-8)
        audio_test = (audio_test - audio_mean) / (audio_std + 1e-8)
        
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

    print(f"Loss: {avg_results[0]:.4f} ± {std_results[0]:.4f}")
    print(f"Accuracy: {avg_results[1]:.4f} ± {std_results[1]:.4f}")
    print(f"Precision: {avg_results[2]:.4f} ± {std_results[2]:.4f}")
    print(f"Recall: {avg_results[3]:.4f} ± {std_results[3]:.4f}")
    print(f"AUC: {avg_results[4]:.4f} ± {std_results[4]:.4f}")

    eval_results_array = np.array(all_eval_results)

    auc_results = eval_results_array[:, 4]  # AUC is at index 4 in your metrics
    best_auc_index = np.argmax(auc_results)
    best_fold_number = best_auc_index + 1
    
    print(f"The best model was found in Fold {best_fold_number} with an AUC of {auc_results[best_auc_index]:.4f}")
    # Construct the file path for the best model
    best_model_path = os.path.join(model_save_dir, f"audio_text_time_model_fold_{best_fold_number}.keras")

    best_model_for_prediction = load_model(best_model_path)
    
    # Save the best model to a new, more descriptive file name
    final_save_path = os.path.join(model_save_dir, "best_audio_text_time_model.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"The best model has been saved to: {final_save_path}")

if __name__ == "__main__":
    train_and_save_multimodal_model(dataset_type="original", remove_short_sentences=False)
    train_and_save_multimodal_model(dataset_type="both", remove_short_sentences=False)