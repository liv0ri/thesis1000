import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Import helper functions and classes from your other files
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
from utils import load_split, pad_sequences_and_times_np
from config import TRAIN_PATH, TEST_PATH, VAL_PATH

# --- CONFIG ---
MAX_SEQUENCE_LENGTH = 50

# --- MODEL DEFINITION ---
def create_audio_time_model(max_sequence_length):
    model_checkpoint = "facebook/wav2vec2-base"
    audio_input = Input(shape=(16000,), dtype=tf.float32)

    # Audio model branch
    audio_features = Wav2VecFeatureExtractor(model_checkpoint)(audio_input)
    audio_output = GlobalAveragePooling1D()(audio_features)
    audio_output = Dropout(0.5)(audio_output)
    audio_model = Model(inputs=audio_input, outputs=audio_output, name='audio_model')

    # Time model branch
    time_stamps = Input(shape=(max_sequence_length, 2), dtype=tf.float32)
    lstm_output = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(time_stamps)
    time_model = Model(inputs=time_stamps, outputs=lstm_output, name='time_model')

    # Concatenate the outputs
    concatenated_output = Concatenate()([audio_model.output, time_model.output])
    combined_output = Dense(1, activation='sigmoid')(concatenated_output)
    
    # Build the combined model
    audio_time_model = Model(inputs=[audio_model.input, time_model.input], outputs=combined_output, name='audio_time_model')
    return audio_time_model

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # It is assumed that the `load_split` function loads all data needed for cross-validation
    all_audios, _, all_times, all_labels = load_split(TRAIN_PATH, load_words=False)
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_results = []
    fold_number = 1

    # Loop through each of the 5 folds
    for train_val_index, test_index in kf.split(all_audios):
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        # Create train/val/test splits for this fold
        audio_train_val = all_audios[train_val_index]
        time_train_val = all_times[train_val_index]
        y_train_val = all_labels[train_val_index]

        audio_test = all_audios[test_index]
        time_test = all_times[test_index]
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
        
        # Re-initialize and compile a new model for each fold
        model = create_audio_time_model(MAX_SEQUENCE_LENGTH)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        # Calculate class weights
        y_train_flat = y_train.flatten()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create a directory to save the models for each fold
        model_save_dir = "models"
        os.makedirs(model_save_dir, exist_ok=True)
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
                  class_weight=class_weight_dict)
        
        model.summary()

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
    
    print("\n✅ Finished all folds.")

    # Find and save the single best model overall
    eval_results_array = np.array(all_eval_results)
    best_accuracy_index = np.argmax(eval_results_array[:, 1])
    best_fold_number = best_accuracy_index + 1
    
    print("\n--- Identifying the Best Model ---")
    print(f"✅ The overall best model was found in Fold {best_fold_number}.")
    
    # Load the best-performing model from its saved location
    best_model_path = os.path.join(model_save_dir, f"audio_time_model_fold_{best_fold_number}.keras")
    best_model_for_prediction = tf.keras.models.load_model(best_model_path)
    
    # Save it to a new, more descriptive filename for final use
    final_save_path = os.path.join(model_save_dir, "best_audio_time_model_overall.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"✅ The best model has been saved to: {final_save_path}")