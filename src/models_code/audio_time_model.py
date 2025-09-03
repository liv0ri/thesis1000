import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from utils import pad_sequences_and_times_np, prepare_audio_data
from config import PROCESSED_DATA_PATH, MAX_SEQUENCE_LENGTH

def create_audio_time_model(max_sequence_length, audio_feature_shape):
    audio_input = Input(shape=audio_feature_shape, dtype=tf.float32, name="audio_input")
    audio_output = Dropout(0.5)(audio_input)
    audio_model = Model(inputs=audio_input, outputs=audio_output, name='audio_model')

    # Time model branch
    time_stamps = Input(shape=(max_sequence_length, 2), dtype=tf.float32, name="time_input")
    lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)
    time_model = Model(inputs=time_stamps, outputs=lstm_output, name='time_model')

    concatenated_output = Concatenate()([audio_model.output, time_model.output])
    combined_output = Dense(1, activation='sigmoid')(concatenated_output)
    
    audio_time_model = Model(inputs=[audio_model.input, time_model.input], outputs=combined_output, name='audio_time_model')
    return audio_time_model

def train_and_save_audio_time_model(dataset_type, remove_short_sentences):
    cache_base_name = f"precomputed_{dataset_type}"
    if remove_short_sentences:
        cache_base_name += "_no_short"
    AUDIO_FEATURES_CACHE_PATH = f"{cache_base_name}_audio_features.npy"
    LABELS_CACHE_PATH = f"{cache_base_name}_labels.npy"
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run the data preparation script first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    # Filter data based on the specified type and length
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
    
    all_times = [d['word_times'] for d in data_points] 
    
    audio_feature_shape = all_audios.shape[1:]

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_number = 1
    all_eval_results = []
    
    model_save_dir = f"models\{dataset_type}"
    if remove_short_sentences:
        model_save_dir += "_no_short"
    os.makedirs(model_save_dir, exist_ok=True)

    # Loop through each of the 5 folds
    for train_val_index, test_index in kf.split(all_audios):
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        # Create train/val/test splits for this fold
        audio_train_val = all_audios[train_val_index]
        time_train_val = [all_times[i] for i in train_val_index]
        y_train_val = all_labels[train_val_index]

        audio_test = all_audios[test_index]
        time_test = [all_times[i] for i in test_index]
        y_test = all_labels[test_index]

        train_size = int(len(audio_train_val) * 0.8)
        audio_train = audio_train_val[:train_size]
        time_train = time_train_val[:train_size]
        y_train = y_train_val[:train_size]
        audio_val = audio_train_val[train_size:]
        time_val = time_train_val[train_size:]
        y_val = y_train_val[train_size:]
        
        # Pad the sequences
        _, time_train_padded = pad_sequences_and_times_np(None, time_train, MAX_SEQUENCE_LENGTH)
        _, time_val_padded = pad_sequences_and_times_np(None, time_val, MAX_SEQUENCE_LENGTH)
        _, time_test_padded = pad_sequences_and_times_np(None, time_test, MAX_SEQUENCE_LENGTH)
        
        time_mean = time_train_padded.mean(axis=(0,1))
        time_std = time_train_padded.std(axis=(0,1))
        time_train_padded = (time_train_padded - time_mean) / (time_std + 1e-8)
        time_val_padded = (time_val_padded - time_mean) / (time_std + 1e-8)
        time_test_padded = (time_test_padded - time_mean) / (time_std + 1e-8)
        
        model = create_audio_time_model(MAX_SEQUENCE_LENGTH, audio_feature_shape)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        # Calculate class weights
        y_train_flat = y_train.flatten()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        checkpoint_path = os.path.join(model_save_dir, f"audio_time_model_fold_{fold_number}.keras")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # Train the model
        model.summary()
        model.fit([audio_train, time_train_padded], y_train,
                  validation_data=([audio_val, time_val_padded], y_val),
                  epochs=25,
                  batch_size=16,
                  shuffle=True,
                  callbacks=[callback, model_checkpoint_callback],
                  class_weight=class_weight_dict,
                  verbose=0)
        # Evaluate on the test data
        eval_results = model.evaluate([audio_test, time_test_padded], y_test)
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
    best_accuracy_index = np.argmax(eval_results_array[:, 1])
    best_fold_number = best_accuracy_index + 1
    
    print(f"The overall best model was found in Fold {best_fold_number}.")
    
    best_model_path = os.path.join(model_save_dir, f"audio_time_model_fold_{best_fold_number}.keras")
    best_model_for_prediction = load_model(best_model_path)
    
    # Save it to a new, more descriptive filename for final use
    final_save_path = os.path.join(model_save_dir, "best_audio_time_model.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"The best model has been saved to: {final_save_path}")

if __name__ == "__main__":
    # train_and_save_audio_time_model(dataset_type="original", remove_short_sentences=False)
    train_and_save_audio_time_model(dataset_type="both", remove_short_sentences=False)
