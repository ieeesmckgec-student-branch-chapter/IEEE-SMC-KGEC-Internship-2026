import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Reshape, Conv2D, MaxPooling2D, Dense, Dropout, 
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, BatchNormalization
)

# =====================================================================
# 1. LOAD EXTRACTED NUMPY ARRAYS
# =====================================================================
print("Loading processed features from disk...")
X_data = np.load("X_features.npy")
y_data = np.load("y_labels.npy")
fold_data = np.load("fold_assignments.npy")

N_MELS = 128            
EXPECTED_FRAMES = 126   
EXPECTED_FEATURES = N_MELS * EXPECTED_FRAMES  

X_flat = X_data.reshape(X_data.shape[0], -1)

# =====================================================================
# SPECAUGMENT DATA AUGMENTATION
# =====================================================================
def augment_spectrogram(melspec):
    augmented = melspec.copy()
    
    t = np.random.randint(0, 15)
    t0 = np.random.randint(0, augmented.shape[1] - t)
    augmented[:, t0:t0 + t] = -80.0
    
    f = np.random.randint(0, 15)
    f0 = np.random.randint(0, augmented.shape[0] - f)
    augmented[f0:f0 + f, :] = -80.0
    
    noise = np.random.normal(0, 0.05, augmented.shape)
    return augmented + noise

def apply_on_the_fly_augmentation(X_raw, y_raw):
    X_augmented = []
    y_augmented = []
    
    for features, label in zip(X_raw, y_raw):
        X_augmented.append(features)
        y_augmented.append(label)
        
        spec = features.reshape(EXPECTED_FRAMES, N_MELS).T
        aug_spec = augment_spectrogram(spec)
        
        X_augmented.append(aug_spec.T.flatten())
        y_augmented.append(label)
            
    return np.array(X_augmented, dtype=np.float32), np.array(y_augmented, dtype=np.float32)

# =====================================================================
# 2. TRANSIENT-AWARE TINYML ARCHITECTURE
# =====================================================================
def create_tinyml_model():
    inputs = Input(shape=(EXPECTED_FEATURES,))
    x = Reshape((EXPECTED_FRAMES, N_MELS, 1))(inputs)
    
    x = Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Reshape((31, 32 * 16))(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    concat = Concatenate()([max_pool, avg_pool])
    
    x = Dense(32, activation='relu')(concat)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model

# =====================================================================
# 3. 10-FOLD CROSS-VALIDATION (UNWEIGHTED STABLE LOSS)
# =====================================================================
fold_accuracies = []
print("\nStarting 10-Fold Cross-Validation...")

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.5, patience=5, min_lr=1e-5
)

for test_fold in range(1, 11):
    X_train_raw_split = X_flat[fold_data != test_fold]
    y_train_raw_split = y_data[fold_data != test_fold]
    
    X_test_raw = X_flat[fold_data == test_fold]
    y_test = y_data[fold_data == test_fold]
    
    X_train_aug_raw, y_train = apply_on_the_fly_augmentation(X_train_raw_split, y_train_raw_split)
    
    mean_train = X_train_aug_raw.mean(axis=0)
    std_train = X_train_aug_raw.std(axis=0) + 1e-8
    
    X_train = (X_train_aug_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train

    model = create_tinyml_model()
    model.fit(
        X_train, y_train, 
        epochs=35, 
        batch_size=32, 
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[lr_scheduler]
    )
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f" - Fold {test_fold:02d} Test Accuracy: {accuracy * 100:.2f}%")
    fold_accuracies.append(accuracy)

print(f"\nMean Accuracy: {np.mean(fold_accuracies) * 100:.2f}% (±{np.std(fold_accuracies) * 100:.2f}%)")

# =====================================================================
# 4. FINAL PRODUCTION MODEL & INT8 EXPORT
# =====================================================================
print("Preparing final production model...")
X_flat_augmented_raw, y_data_augmented = apply_on_the_fly_augmentation(X_flat, y_data)

X_mean_final = X_flat_augmented_raw.mean(axis=0)
X_std_final = X_flat_augmented_raw.std(axis=0) + 1e-8
X_flat_normalized = (X_flat_augmented_raw - X_mean_final) / X_std_final

np.save("mean.npy", X_mean_final)
np.save("std.npy", X_std_final)

final_model = create_tinyml_model()
final_model.fit(
    X_flat_normalized, y_data_augmented, 
    epochs=40, 
    batch_size=32, 
    verbose=0,
    callbacks=[lr_scheduler]
)

print("\nConverting model to FULL INTEGER INT8 TensorFlow Lite...")
np.random.seed(42)
sample_indices = np.random.choice(X_flat_normalized.shape[0], size=200, replace=False)

def representative_data_gen():
    for idx in sample_indices:
        sample = np.expand_dims(X_flat_normalized[idx], axis=0).astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_int8 = converter.convert()

with open("gunshot_detector.tflite", "wb") as f:
    f.write(tflite_model_int8)

print(f"[SUCCESS] Exported 'gunshot_detector.tflite' ({len(tflite_model_int8)/1024:.2f} KB)")