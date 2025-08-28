import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from wav2vec_feature_extractor import Wav2VecFeatureExtractor
PROCESSED_DATA_PATH = "processed_data.pkl"

# --- MODEL DEFINITION ---
def create_audio_model():
    model_checkpoint = "facebook/wav2vec2-base"
    input_values = Input(shape=(16000,), dtype=tf.float32, name="audio_input")
    wav2vec_features = Wav2VecFeatureExtractor(model_checkpoint)(input_values)
    audio_model_output = GlobalAveragePooling1D()(wav2vec_features)
    audio_model_output = Dropout(0.5)(audio_model_output)
    output = Dense(1, activation='sigmoid')(audio_model_output)
    audio_model = Model(inputs=input_values, outputs=output)
    audio_model.summary() # Corrected summary call
    return audio_model

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Load the entire dataset from the single processed file
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run the data preparation script first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    all_audios = np.array([d['audio'] for d in data_points])
    all_labels = np.array([1 if d['label'] == 'dementia' else 0 for d in data_points])
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_results = []
    fold_number = 1
    
    # Create the models directory if it doesn't exist
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Loop through each of the 5 folds
    for train_val_index, test_index in kf.split(all_audios):
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        # Create train/val/test splits for this fold
        audio_train_val = all_audios[train_val_index]
        y_train_val = all_labels[train_val_index]

        audio_test = all_audios[test_index]
        y_test = all_labels[test_index]

        train_size = int(len(audio_train_val) * 0.8)
        audio_train = audio_train_val[:train_size]
        y_train = y_train_val[:train_size]
        audio_val = audio_train_val[train_size:]
        y_val = y_train_val[train_size:]
        
        # Re-initialize and compile a new model for each fold
        model = create_audio_model()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        
        # Calculate class weights
        y_train_flat = y_train.flatten()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Early stopping callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint_path = os.path.join(model_save_dir, f"audio_model_fold_{fold_number}.keras")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # Train the model
        model.fit(audio_train, y_train,
                  validation_data=(audio_val, y_val),
                  epochs=25,
                  batch_size=16,
                  shuffle=True,
                  callbacks=[callback, model_checkpoint_callback],
                  class_weight=class_weight_dict,
                  verbose=0) 
        
        # Evaluate on the test data
        eval_results = model.evaluate(audio_test, y_test)
        all_eval_results.append(eval_results)
        
        print(f"Fold {fold_number} Test Results: {eval_results}")
        fold_number += 1

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
        
    # Load the best-performing model from its saved location
    best_model_path = os.path.join(model_save_dir, f"audio_model_fold_{best_fold_number}.keras")
    best_model_for_prediction = load_model(best_model_path)
    
    final_save_path = os.path.join(model_save_dir, "best_model_audio_overall.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"✅ The best model has been saved to: {final_save_path}")
