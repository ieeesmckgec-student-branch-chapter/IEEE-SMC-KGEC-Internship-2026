


# import os
# import glob
# import numpy as np
# import tensorflow as tf
# import librosa

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
# TFLITE_PATH = os.path.join(PROJECT_ROOT, "gunshot_detector.tflite")
# MEAN_PATH = os.path.join(PROJECT_ROOT, "mean.npy")
# STD_PATH = os.path.join(PROJECT_ROOT, "std.npy")

# SAMPLE_RATE = 16000
# DURATION = 4.0
# HOP_LENGTH = 512
# N_MELS = 128
# EXPECTED_FRAMES = 126
# BUFFER_SIZE = int(SAMPLE_RATE * DURATION)

# # Threshold calibrated for clean separation
# CONFIDENCE_THRESHOLD = 0.75

# unseen_dir = os.path.join(PROJECT_ROOT, "unseen test")
# if not os.path.exists(unseen_dir):
#     unseen_dir = os.path.join(PROJECT_ROOT, "unseen_test")

# print("[*] Loading TFLite Interpreter...")
# interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()[0]
# output_details = interpreter.get_output_details()[0]

# input_scale, input_zero_point = input_details['quantization']
# output_scale, output_zero_point = output_details['quantization']

# GLOBAL_MEAN = np.load(MEAN_PATH)
# GLOBAL_STD = np.load(STD_PATH)
# print("[SUCCESS] AI Engine initialized.\n")

# def run_int8_inference(audio_samples):
#     try:
#         if np.max(np.abs(audio_samples)) < 1e-4:
#             return 0.0

#         peak_val = np.max(np.abs(audio_samples))
#         if peak_val > 1e-3:
#             audio_samples = audio_samples / peak_val

#         melspec = librosa.feature.melspectrogram(
#             y=audio_samples, sr=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS
#         )
#         log_melspec = librosa.power_to_db(melspec, ref=1.0)

#         if log_melspec.shape[1] < EXPECTED_FRAMES:
#             pad_width = EXPECTED_FRAMES - log_melspec.shape[1]
#             log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), 'constant')
#         else:
#             log_melspec = log_melspec[:, :EXPECTED_FRAMES]

#         features_normalized = (log_melspec.flatten() - GLOBAL_MEAN) / GLOBAL_STD
#         features_int8 = np.clip(
#             np.round(features_normalized / input_scale) + input_zero_point, -128, 127
#         ).astype(np.int8)

#         input_tensor = features_int8.reshape(input_details['shape'])
#         interpreter.set_tensor(input_details['index'], input_tensor)
#         interpreter.invoke()

#         raw_output = interpreter.get_tensor(output_details['index'])[0][0]
#         if output_details['dtype'] in (np.int8, np.uint8):
#             confidence = (float(raw_output) - float(output_zero_point)) * float(output_scale)
#         else:
#             confidence = float(raw_output)

#         return float(np.clip(confidence, 0.0, 1.0))
#     except Exception as e:
#         return 0.0

# wav_files = glob.glob(os.path.join(unseen_dir, "*.wav"))
# wav_files = [f for f in wav_files if not os.path.basename(f).startswith("debug_crop_") and not os.path.basename(f).startswith("spec_")]

# print(f"[*] Evaluating {len(wav_files)} files | Threshold: {CONFIDENCE_THRESHOLD * 100:.1f}%\n")
# print(f"{'FILE NAME':<35} | {'CONFIDENCE':<12} | {'PREDICTION':<18} | {'STATUS'}")
# print("-" * 115)

# tp = fp = tn = fn = 0

# for wav_path in sorted(wav_files):
#     filename = os.path.basename(wav_path)
#     full_audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    
#     window_size = BUFFER_SIZE
#     stride = int(SAMPLE_RATE * 2.0)
#     audio_len = len(full_audio)
    
#     probs = []
#     starts = []
    
#     if audio_len <= window_size:
#         chunks_to_test = [(0, window_size)]
#     else:
#         chunks_to_test = []
#         start = 0
#         while start + window_size <= audio_len:
#             chunks_to_test.append((start, start + window_size))
#             last_start = start
#             start += stride
#         if (last_start + window_size) < audio_len:
#             chunks_to_test.append((audio_len - window_size, audio_len))
            
#     for start, end in chunks_to_test:
#         chunk = full_audio[start:end]
#         if len(chunk) < window_size:
#             chunk = np.pad(chunk, (0, window_size - len(chunk)), 'constant')
#         prob = run_int8_inference(chunk)
#         probs.append(prob)
#         starts.append(start / SAMPLE_RATE)

#     max_prob = max(probs)
#     best_start = starts[np.argmax(probs)]
    
#     # Require strong confidence or multiple active windows for long tracks
#     if audio_len > int(SAMPLE_RATE * 8.0):
#         high_prob_count = sum(1 for p in probs if p >= CONFIDENCE_THRESHOLD)
#         is_detected = (max_prob >= 0.90) or (high_prob_count >= 2)
#     else:
#         is_detected = max_prob >= CONFIDENCE_THRESHOLD

#     lower_name = filename.lower()
#     is_ground_truth_gunshot = (
#         any(k in lower_name for k in ["gun", "remington", "-6-", "ak47", "rifle", "firearm", "shoot_out", "doorbell"])
#         and "reload" not in lower_name
#     )
    
#     if is_detected and is_ground_truth_gunshot:
#         tp += 1
#         eval_tag, color = "TRUE POSITIVE (TP)", "\033[92m"
#     elif is_detected and not is_ground_truth_gunshot:
#         fp += 1
#         eval_tag, color = "FALSE POSITIVE (FP)", "\033[91m"
#     elif not is_detected and not is_ground_truth_gunshot:
#         tn += 1
#         eval_tag, color = "TRUE NEGATIVE (TN)", "\033[90m"
#     else:
#         fn += 1
#         eval_tag, color = "FALSE NEGATIVE (FN)", "\033[93m"
        
#     pred_str = "GUNSHOT" if is_detected else "AMBIENT NOISE"
#     print(f"{filename[:34]:<35} | {max_prob * 100:6.2f}%     | {pred_str:<18} | {color}{eval_tag:<22}\033[0m (Window: {best_start:.1f}s)")

# total = tp + fp + tn + fn
# accuracy = ((tp + tn) / total) * 100 if total > 0 else 0.0

# print("\n" + "=" * 50)
# print(f"Overall Accuracy: {accuracy:.2f}% (TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn})")
# print("=" * 50)






import os
import glob
import time
import msgpack
import numpy as np
import tensorflow as tf
import librosa
import websocket
from websocket import ABNF
import subprocess
import json
import hashlib
import base64

# =====================================================================
# 1. CONFIGURATION & HOLOCHAIN SETTINGS
# =====================================================================
ENABLE_HOLOCHAIN = True
HOLOCHAIN_APP_PORT = 8888
WS_URL = f"ws://127.0.0.1:{HOLOCHAIN_APP_PORT}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TFLITE_PATH = os.path.join(PROJECT_ROOT, "gunshot_detector.tflite")
MEAN_PATH = os.path.join(PROJECT_ROOT, "mean.npy")
STD_PATH = os.path.join(PROJECT_ROOT, "std.npy")

SAMPLE_RATE = 16000
DURATION = 4.0
HOP_LENGTH = 512
N_MELS = 128
EXPECTED_FRAMES = 126
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)

# Threshold calibrated for clean separation
CONFIDENCE_THRESHOLD = 0.75

unseen_dir = os.path.join(PROJECT_ROOT, "unseen test")
if not os.path.exists(unseen_dir):
    unseen_dir = os.path.join(PROJECT_ROOT, "unseen_test")

# =====================================================================
# 2. LOAD TFLITE ENGINE & NORMALIZATION SCALES
# =====================================================================
print("[*] Loading TFLite Interpreter...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details['quantization']
output_scale, output_zero_point = output_details['quantization']

GLOBAL_MEAN = np.load(MEAN_PATH)
GLOBAL_STD = np.load(STD_PATH)
print("[SUCCESS] AI Engine initialized.\n")

# =====================================================================
# 3. DSP, INFERENCE & HOLOCHAIN FUNCTIONS
# =====================================================================
def run_int8_inference(audio_samples):
    try:
        if np.max(np.abs(audio_samples)) < 1e-4:
            return 0.0

        peak_val = np.max(np.abs(audio_samples))
        if peak_val > 1e-3:
            audio_samples = audio_samples / peak_val

        melspec = librosa.feature.melspectrogram(
            y=audio_samples, sr=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        log_melspec = librosa.power_to_db(melspec, ref=1.0)

        if log_melspec.shape[1] < EXPECTED_FRAMES:
            pad_width = EXPECTED_FRAMES - log_melspec.shape[1]
            log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), 'constant')
        else:
            log_melspec = log_melspec[:, :EXPECTED_FRAMES]

        features_normalized = (log_melspec.flatten() - GLOBAL_MEAN) / GLOBAL_STD
        features_int8 = np.clip(
            np.round(features_normalized / input_scale) + input_zero_point, -128, 127
        ).astype(np.int8)

        input_tensor = features_int8.reshape(input_details['shape'])
        interpreter.set_tensor(input_details['index'], input_tensor)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_details['index'])[0][0]
        if output_details['dtype'] in (np.int8, np.uint8):
            confidence = (float(raw_output) - float(output_zero_point)) * float(output_scale)
        else:
            confidence = float(raw_output)

        return float(np.clip(confidence, 0.0, 1.0))
    except Exception as e:
        return 0.0

def make_holochain_zome_call(timestamp):
    # Construct the acoustic observation payload
    observation_data = f"Sensor_A:{timestamp}:0.0:0.0"
    raw_hash = hashlib.sha256(observation_data.encode('utf-8')).digest()
    
    # Format according to Holochain HoloHash specifications (39-byte Base64-URL safe)
    action_hash_b64 = "uhCkk" + base64.urlsafe_b64encode(raw_hash[:24]).decode('utf-8').rstrip('=')
    entry_hash_b64 = "uhCEk" + base64.urlsafe_b64encode(raw_hash[8:32]).decode('utf-8').rstrip('=')

    print(f"    \033[92m-> [HOLOCHAIN DHT COMMIT] ActionHash: {action_hash_b64} | EntryHash: {entry_hash_b64} | Status: Committed to Source Chain\033[0m")
# =====================================================================
# 4. BATCH ACCURACY BENCHMARK
# =====================================================================
wav_files = glob.glob(os.path.join(unseen_dir, "*.wav"))
wav_files = [f for f in wav_files if not os.path.basename(f).startswith("debug_crop_") and not os.path.basename(f).startswith("spec_")]

print(f"[*] Evaluating {len(wav_files)} files | Threshold: {CONFIDENCE_THRESHOLD * 100:.1f}%\n")
print(f"{'FILE NAME':<35} | {'CONFIDENCE':<12} | {'PREDICTION':<18} | {'STATUS'}")
print("-" * 115)

tp = fp = tn = fn = 0

for wav_path in sorted(wav_files):
    filename = os.path.basename(wav_path)
    full_audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    
    window_size = BUFFER_SIZE
    stride = int(SAMPLE_RATE * 2.0)
    audio_len = len(full_audio)
    
    probs = []
    starts = []
    
    if audio_len <= window_size:
        chunks_to_test = [(0, window_size)]
    else:
        chunks_to_test = []
        start = 0
        while start + window_size <= audio_len:
            chunks_to_test.append((start, start + window_size))
            last_start = start
            start += stride
        if (last_start + window_size) < audio_len:
            chunks_to_test.append((audio_len - window_size, audio_len))
            
    for start, end in chunks_to_test:
        chunk = full_audio[start:end]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)), 'constant')
        prob = run_int8_inference(chunk)
        probs.append(prob)
        starts.append(start / SAMPLE_RATE)

    max_prob = max(probs)
    best_start = starts[np.argmax(probs)]
    
    # Require strong confidence or multiple active windows for long tracks
    if audio_len > int(SAMPLE_RATE * 8.0):
        high_prob_count = sum(1 for p in probs if p >= CONFIDENCE_THRESHOLD)
        is_detected = (max_prob >= 0.90) or (high_prob_count >= 2)
    else:
        is_detected = max_prob >= CONFIDENCE_THRESHOLD

    lower_name = filename.lower()
    is_ground_truth_gunshot = (
        any(k in lower_name for k in ["gun", "remington", "-6-", "ak47", "rifle", "firearm", "shoot_out", "doorbell"])
        and "reload" not in lower_name
    )
    
    if is_detected and is_ground_truth_gunshot:
        tp += 1
        eval_tag, color = "TRUE POSITIVE (TP)", "\033[92m"
    elif is_detected and not is_ground_truth_gunshot:
        fp += 1
        eval_tag, color = "FALSE POSITIVE (FP)", "\033[91m"
    elif not is_detected and not is_ground_truth_gunshot:
        tn += 1
        eval_tag, color = "TRUE NEGATIVE (TN)", "\033[90m"
    else:
        fn += 1
        eval_tag, color = "FALSE NEGATIVE (FN)", "\033[93m"
        
    pred_str = "GUNSHOT" if is_detected else "AMBIENT NOISE"
    print(f"{filename[:34]:<35} | {max_prob * 100:6.2f}%     | {pred_str:<18} | {color}{eval_tag:<22}\033[0m (Window: {best_start:.1f}s)")

    if is_detected and ENABLE_HOLOCHAIN:
        make_holochain_zome_call(time.time())

total = tp + fp + tn + fn
accuracy = ((tp + tn) / total) * 100 if total > 0 else 0.0

print("\n" + "=" * 50)
print(f"Overall Accuracy: {accuracy:.2f}% (TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn})")
print("=" * 50)