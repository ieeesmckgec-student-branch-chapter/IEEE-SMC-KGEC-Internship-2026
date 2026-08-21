import os
import glob
import pandas as pd
import numpy as np
import librosa

# =====================================================================
# CONFIGURATION & DIRECTORY SETUP
# =====================================================================
CSV_PATH = os.path.join("UrbanSound8K", "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join("UrbanSound8K", "audio")

CUSTOM_GUNSHOTS_DIR = os.path.join("custom_data", "gunshots")
CUSTOM_NEGATIVES_DIR = os.path.join("custom_data", "negatives")

TARGET_SR = 16000     # Target sample rate (16kHz)
DURATION = 4.0        # Window duration in seconds
STRIDE = 2.0          # 50% overlap stride for slicing long files
N_MELS = 128          # 128 Mel frequency bands
HOP_LENGTH = 512      # 32ms temporal resolution
EXPECTED_FRAMES = 126 # 126 time frames for 64,000 samples @ hop=512

if not os.path.exists(CSV_PATH):
    print(f"Error: Metadata file not found at {CSV_PATH}")
    exit(1)

# =====================================================================
# 1. CORE SPECTROGRAM PROCESSOR
# =====================================================================
def audio_chunk_to_melspec(audio_chunk):
    """Converts a 4.0s normalized audio chunk directly into a 128x126 Log-Mel matrix."""
    # Enforce exact waveform size (64,000 samples)
    total_samples = int(TARGET_SR * DURATION)
    if len(audio_chunk) < total_samples:
        audio_chunk = np.pad(audio_chunk, (0, total_samples - len(audio_chunk)), 'constant')
    else:
        audio_chunk = audio_chunk[:total_samples]
        
    melspec = librosa.feature.melspectrogram(
        y=audio_chunk, sr=TARGET_SR, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_melspec = librosa.power_to_db(melspec, ref=1.0)
    
    # Enforce exact frame width (126 columns)
    if log_melspec.shape[1] < EXPECTED_FRAMES:
        pad_width = EXPECTED_FRAMES - log_melspec.shape[1]
        log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), 'constant')
    else:
        log_melspec = log_melspec[:, :EXPECTED_FRAMES]
        
    return log_melspec

# =====================================================================
# 2. BALANCED SAMPLING (URBANSOUND8K)
# =====================================================================
print("Loading UrbanSound8K metadata...")
metadata = pd.read_csv(CSV_PATH)
print("Performing within-fold balanced sampling...")

balanced_records = []
for fold_id in range(1, 11):
    fold_subset = metadata[metadata['fold'] == fold_id]
    
    fold_gunshots = fold_subset[fold_subset['classID'] == 6].copy()
    fold_gunshots['label'] = 1
    n_gunshots = len(fold_gunshots)
    
    fold_non_gunshots = fold_subset[fold_subset['classID'] != 6].copy()
    if n_gunshots == 0:
        continue
        
    if len(fold_non_gunshots) >= n_gunshots:
        sampled_non_gunshots = fold_non_gunshots.sample(n=n_gunshots, random_state=42)
    else:
        sampled_non_gunshots = fold_non_gunshots
        
    sampled_non_gunshots['label'] = 0
    balanced_records.append(fold_gunshots)
    balanced_records.append(sampled_non_gunshots)

balanced_metadata = pd.concat(balanced_records).reset_index(drop=True)
print(f"UrbanSound8K Balanced! Collected {len(balanced_metadata)} records across 10 folds.")

# =====================================================================
# 3. EXTRACTION LOOP
# =====================================================================
X_features = []
y_labels = []
folds = []
failed_files = []

print("\nStarting UrbanSound8K feature extraction...")
for index, row in balanced_metadata.iterrows():
    file_path = os.path.join(AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
    try:
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True, duration=DURATION)
        melspec = audio_chunk_to_melspec(audio)
        X_features.append(melspec)
        y_labels.append(row['label'])
        folds.append(row['fold'])
    except Exception as e:
        failed_files.append((file_path, row['slice_file_name']))
        
    if (index + 1) % 200 == 0:
        print(f"Processed {index + 1}/{len(balanced_metadata)} UrbanSound8K files...")

def load_custom_folder_with_slicing(folder_path, label):
    """
    Loads full-length custom audio files and slices them into 4.0s windows with 2.0s stride.
    Captures explosions, mortars, and transients regardless of where they appear in the file.
    """
    if not os.path.exists(folder_path):
        print(f"[*] Folder '{folder_path}' not found. Skipping.")
        return
        
    audio_extensions = ("*.wav", "*.ogg", "*.mp3", "*.flac")
    files = []
    for ext in audio_extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    print(f"\n[*] Processing {len(files)} custom files from '{folder_path}' (Label: {label})...")
    window_samples = int(TARGET_SR * DURATION)
    stride_samples = int(TARGET_SR * STRIDE)
    
    extracted_slices = 0
    for i, fpath in enumerate(files):
        try:
            # Load full audio file without duration cutoff
            full_audio, _ = librosa.load(fpath, sr=TARGET_SR, mono=True)
            audio_len = len(full_audio)
            
            if audio_len <= window_samples:
                chunks = [full_audio]
            else:
                chunks = []
                start = 0
                while start + window_samples <= audio_len:
                    chunks.append(full_audio[start:start + window_samples])
                    last_start = start
                    start += stride_samples
                if (last_start + window_samples) < audio_len:
                    chunks.append(full_audio[audio_len - window_samples:audio_len])
                    
            for chunk in chunks:
                melspec = audio_chunk_to_melspec(chunk)
                X_features.append(melspec)
                y_labels.append(label)
                folds.append((extracted_slices % 10) + 1)  # Spread slices across all 10 folds
                extracted_slices += 1
        except Exception as e:
            failed_files.append((fpath, os.path.basename(fpath)))
            
    print(f" -> Extracted {extracted_slices} total 4.0s slices from '{folder_path}'.")

# Ingest custom positive (gunshots) and negative (fireworks/mortars/chaos) datasets
load_custom_folder_with_slicing(CUSTOM_GUNSHOTS_DIR, label=1)
load_custom_folder_with_slicing(CUSTOM_NEGATIVES_DIR, label=0)

# =====================================================================
# 4. SAVE EXTRACTED NUMPY DATASETS
# =====================================================================
X_data = np.array(X_features, dtype=np.float32)
y_data = np.array(y_labels, dtype=np.int32)
fold_data = np.array(folds, dtype=np.int32)

print("\n==========================================")
print("              EXTRACTION SUMMARY           ")
print("==========================================")
print(f"Total extracted samples:  {len(X_data)}")
print(f" - Gunshot Samples (1):   {np.sum(y_data == 1)}")
print(f" - Non-Gunshot Noise (0): {np.sum(y_data == 0)}")
print(f"Failed extractions:       {len(failed_files)}")
print(f"Features matrix shape:    {X_data.shape} (Expected: [Samples, {N_MELS}, {EXPECTED_FRAMES}])")
print("==========================================")

np.save("X_features.npy", X_data)
np.save("y_labels.npy", y_data)
np.save("fold_assignments.npy", fold_data)
print("\nNumPy features saved successfully to workspace disk.")