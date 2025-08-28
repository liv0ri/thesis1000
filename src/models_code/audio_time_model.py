import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Import helper functions and classes from your other files
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from utils import pad_sequences_and_times_np

PROCESSED_DATA_PATH = "processed_data.pkl"
MAX_SEQUENCE_LENGTH = 50

# --- MODEL DEFINITION ---
def create_audio_time_model(max_sequence_length):
    model_checkpoint = "facebook/wav2vec2-base"
    audio_input = Input(shape=(16000,), dtype=tf.float32, name="audio_input")

    # Audio model branch
    audio_features = Wav2VecFeatureExtractor(model_checkpoint)(audio_input)
    audio_output = GlobalAveragePooling1D()(audio_features)
    audio_output = Dropout(0.5)(audio_output)
    audio_model = Model(inputs=audio_input, outputs=audio_output, name='audio_model')

    # Time model branch
    time_stamps = Input(shape=(max_sequence_length, 2), dtype=tf.float32, name="time_input")
    lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)
    time_model = Model(inputs=time_stamps, outputs=lstm_output, name='time_model')

    concatenated_output = Concatenate()([audio_model.output, time_model.output])
    combined_output = Dense(1, activation='sigmoid')(concatenated_output)
    
    audio_time_model = Model(inputs=[audio_model.input, time_model.input], outputs=combined_output, name='audio_time_model')
    return audio_time_model

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run the data preparation script first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    all_audios = np.array([d['audio'] for d in data_points])
    all_times = [d['word_times'] for d in data_points]
    all_labels = np.array([1 if d['label'] == 'dementia' else 0 for d in data_points])
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_results = []
    fold_number = 1

    model_save_dir = "models"
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
        
        model = create_audio_time_model(MAX_SEQUENCE_LENGTH)
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

    print("\n--- Final Results (Mean ± Std Dev) ---")
    print(f"Loss: {avg_results[0]:.4f} ± {std_results[0]:.4f}")
    print(f"Accuracy: {avg_results[1]:.4f} ± {std_results[1]:.4f}")
    print(f"Precision: {avg_results[2]:.4f} ± {std_results[2]:.4f}")
    print(f"Recall: {avg_results[3]:.4f} ± {std_results[3]:.4f}")
    print(f"AUC: {avg_results[4]:.4f} ± {std_results[4]:.4f}")
    
    eval_results_array = np.array(all_eval_results)
    best_accuracy_index = np.argmax(eval_results_array[:, 1])
    best_fold_number = best_accuracy_index + 1
    
    print(f"✅ The overall best model was found in Fold {best_fold_number}.")
    
    best_model_path = os.path.join(model_save_dir, f"audio_time_model_fold_{best_fold_number}.keras")
    best_model_for_prediction = load_model(best_model_path)
    
    # Save it to a new, more descriptive filename for final use
    final_save_path = os.path.join(model_save_dir, "best_audio_time_model_overall.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"✅ The best model has been saved to: {final_save_path}")