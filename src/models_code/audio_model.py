import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.utils.class_weight import compute_class_weight
from utils import prepare_audio_data
from sklearn.model_selection import StratifiedKFold, train_test_split
from config import PROCESSED_DATA_PATH

def create_audio_model(input_shape):
    input_features = Input(shape=input_shape, dtype=tf.float32, name="audio_input")
    x = Dropout(0.5)(input_features)
    output = Dense(1, activation='sigmoid')(x)
    audio_model = Model(inputs=input_features, outputs=output)
    audio_model.summary()
    return audio_model

def train_and_save_model(dataset_type, remove_short_sentences):
    cache_base_name = f"precomputed_{dataset_type}"
    if remove_short_sentences:
        cache_base_name += "_no_short"
    FEATURES_CACHE_PATH = f"{cache_base_name}_audio_features.npy"
    LABELS_CACHE_PATH = f"{cache_base_name}_labels.npy"

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run the data preparation script first.")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_points = pickle.load(f)

    # Filter data based on the chosen dataset_type
    if dataset_type == "original":
        data_points = [d for d in data_points if d['source_type'] == 'original']
    elif dataset_type == "augmented":
        data_points = [d for d in data_points if d['source_type'] == 'text_augmented']
    if remove_short_sentences:
        data_points = [d for d in data_points if len(d['words']) >= 5]

    if os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(LABELS_CACHE_PATH):
        all_audios = np.load(FEATURES_CACHE_PATH)
        all_labels = np.load(LABELS_CACHE_PATH)
    else:
        all_audios, all_labels = prepare_audio_data(data_points, FEATURES_CACHE_PATH, LABELS_CACHE_PATH)

    # Get the input shape for the model from the pre-computed features
    input_shape = all_audios.shape[1:]

    # Cross-validation setup
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_results = []
    fold_number = 1
    
    model_save_dir = f"models\{dataset_type}"
    if remove_short_sentences:
        model_save_dir += "_no_short"
    os.makedirs(model_save_dir, exist_ok=True)

    # Loop through each of the 5 folds
    for train_val_index, test_index in kf.split(all_audios, all_labels):
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        # Create train/val/test splits for this fold
        audio_train_val = all_audios[train_val_index]
        y_train_val = all_labels[train_val_index]

        audio_train, audio_val, y_train, y_val = train_test_split(
            audio_train_val,
            y_train_val,
            test_size=0.2,
            stratify=y_train_val,
            random_state=42
        )

        audio_test = all_audios[test_index]
        y_test = all_labels[test_index]

        audio_mean = audio_train.mean(axis=0)
        audio_std = audio_train.std(axis=0)
        audio_train = (audio_train - audio_mean) / (audio_std + 1e-8)
        audio_val = (audio_val - audio_mean) / (audio_std + 1e-8)
        audio_test = (audio_test - audio_mean) / (audio_std + 1e-8)

        # Re-initialize and compile a new model for each fold
        model = create_audio_model(input_shape)
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
    
    print(f"The overall best model was found in Fold {best_fold_number}.")
    
    # Load the best-performing model from its saved location
    best_model_path = os.path.join(model_save_dir, f"audio_model_fold_{best_fold_number}.keras")
    best_model_for_prediction = load_model(best_model_path)
    
    final_save_path = os.path.join(model_save_dir, f"best_audio_model.keras")
    best_model_for_prediction.save(final_save_path)
    print(f"The best model has been saved to: {final_save_path}")
if __name__ == "__main__":
    train_and_save_model(dataset_type="original", remove_short_sentences=False)
    train_and_save_model(dataset_type="both", remove_short_sentences=False)
