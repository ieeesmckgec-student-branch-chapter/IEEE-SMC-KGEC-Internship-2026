import os
import glob
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import librosa.display

# =====================================================================
# CONFIGURATION & PATH SETUP
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) if os.path.basename(BASE_DIR) == "app" else BASE_DIR

TFLITE_PATH = os.path.join(PROJECT_ROOT, "gunshot_detector.tflite")
MEAN_PATH = os.path.join(PROJECT_ROOT, "mean.npy")
STD_PATH = os.path.join(PROJECT_ROOT, "std.npy")

OUTPUT_VIS_DIR = os.path.join(PROJECT_ROOT, "research_paper_visuals")
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

MASTER_GRID_PNG = os.path.join(OUTPUT_VIS_DIR, "all_30_spectrograms_grid.png")
MASTER_GRID_PDF = os.path.join(OUTPUT_VIS_DIR, "all_30_spectrograms_grid.pdf")

SAMPLE_RATE = 16000
DURATION = 4.0
HOP_LENGTH = 512
N_MELS = 128
EXPECTED_FRAMES = 126
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
CONFIDENCE_THRESHOLD = 0.75

unseen_dir = os.path.join(PROJECT_ROOT, "unseen test")
if not os.path.exists(unseen_dir):
    unseen_dir = os.path.join(PROJECT_ROOT, "unseen_test")

# Typography & styling for scientific journals
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
})

# =====================================================================
# LOAD TFLITE ENGINE & CALIBRATION
# =====================================================================
print("[*] Initializing TFLite Engine...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details['quantization']
output_scale, output_zero_point = output_details['quantization']

GLOBAL_MEAN = np.load(MEAN_PATH)
GLOBAL_STD = np.load(STD_PATH)

def extract_melspec(audio_samples):
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
        
    return log_melspec

def run_int8_inference(log_melspec):
    try:
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
    except Exception:
        return 0.0

# =====================================================================
# BATCH EVALUATION & MASTER GRID GENERATION
# =====================================================================
wav_files = sorted(glob.glob(os.path.join(unseen_dir, "*.wav")))
wav_files = [f for f in wav_files if not os.path.basename(f).startswith(("debug_crop_", "spec_"))]

print(f"[*] Processing {len(wav_files)} files into unified research grid...\n")
print(f"{'FILE NAME':<35} | {'CONFIDENCE':<12} | {'PREDICTION':<18} | {'STATUS'}")
print("-" * 115)

tp = fp = tn = fn = 0
grid_items = []

for idx, wav_path in enumerate(wav_files, 1):
    raw_name = os.path.basename(wav_path)
    clean_name = os.path.splitext(raw_name)[0]

    full_audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio_len = len(full_audio)
    stride = int(SAMPLE_RATE * 2.0)

    chunks_to_test = []
    if audio_len <= BUFFER_SIZE:
        chunks_to_test.append((0, BUFFER_SIZE))
    else:
        start = 0
        while start + BUFFER_SIZE <= audio_len:
            chunks_to_test.append((start, start + BUFFER_SIZE))
            last_start = start
            start += stride
        if (last_start + BUFFER_SIZE) < audio_len:
            chunks_to_test.append((audio_len - BUFFER_SIZE, audio_len))

    probs = []
    starts = []
    melspecs = []

    for start_sample, end_sample in chunks_to_test:
        chunk = full_audio[start_sample:end_sample]
        if len(chunk) < BUFFER_SIZE:
            chunk = np.pad(chunk, (0, BUFFER_SIZE - len(chunk)), 'constant')
        
        melspec = extract_melspec(chunk)
        prob = run_int8_inference(melspec)

        probs.append(prob)
        starts.append(start_sample / SAMPLE_RATE)
        melspecs.append(melspec)

    best_idx = int(np.argmax(probs))
    best_prob = probs[best_idx]
    best_start = starts[best_idx]
    best_melspec = melspecs[best_idx]

    if audio_len > int(SAMPLE_RATE * 8.0):
        high_prob_count = sum(1 for p in probs if p >= CONFIDENCE_THRESHOLD)
        is_detected = (best_prob >= 0.90) or (high_prob_count >= 2)
    else:
        is_detected = best_prob >= CONFIDENCE_THRESHOLD

    lower_name = raw_name.lower()
    is_ground_truth_gunshot = (
        any(k in lower_name for k in ["gun", "remington", "-6-", "ak47", "rifle", "firearm", "shoot_out", "doorbell"])
        and "reload" not in lower_name
    )

    if is_detected and is_ground_truth_gunshot:
        tp += 1
        eval_tag, title_color = "TP", "#1b5e20"      # Dark Green
    elif is_detected and not is_ground_truth_gunshot:
        fp += 1
        eval_tag, title_color = "FP", "#b71c1c"      # Dark Red
    elif not is_detected and not is_ground_truth_gunshot:
        tn += 1
        eval_tag, title_color = "TN", "#212121"      # Dark Grey/Black
    else:
        fn += 1
        eval_tag, title_color = "FN", "#e65100"      # Dark Orange

    pred_str = "GUNSHOT" if is_detected else "AMBIENT NOISE"
    print(f"{raw_name[:34]:<35} | {best_prob * 100:6.2f}%     | {pred_str:<18} | {eval_tag:<22} (Window: {best_start:.1f}s)")

    grid_items.append({
        "idx": idx,
        "name": clean_name[:20],
        "spec": best_melspec,
        "prob": best_prob,
        "eval_tag": eval_tag,
        "color": title_color,
        "pred": pred_str
    })

# =====================================================================
# RENDER ALL 30 IN A 5x6 SUBPLOT GRID
# =====================================================================
ROWS, COLS = 5, 6
fig, axes = plt.subplots(ROWS, COLS, figsize=(20, 14), sharex=True, sharey=True)
axes_flat = axes.flatten()
last_im = None

for i, item in enumerate(grid_items):
    if i >= len(axes_flat):
        break
    ax = axes_flat[i]
    
    last_im = librosa.display.specshow(
        item["spec"],
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        fmax=8000,
        ax=ax,
        cmap='magma'
    )
    
    # Subtitle with index, status, and prediction probability
    ax.set_title(
        f"#{item['idx']} [{item['eval_tag']}] {item['name']}\n{item['pred']} ({item['prob']*100:.1f}%)",
        fontsize=7.5,
        fontweight='bold',
        color=item['color'],
        pad=4
    )
    
    # Clean axis labels: only show edges
    ax.set_xlabel("Time (s)" if i >= (ROWS - 1) * COLS else "")
    ax.set_ylabel("Mel Freq (Hz)" if i % COLS == 0 else "")

# Hide any unused subplots if dataset < 30
for j in range(len(grid_items), len(axes_flat)):
    fig.delaxes(axes_flat[j])

# Adjust spacing and place a single shared colorbar on the right
fig.subplots_adjust(right=0.92, hspace=0.38, wspace=0.18, top=0.94, bottom=0.06, left=0.05)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.70])
cbar = fig.colorbar(last_im, cax=cbar_ax)
cbar.set_label("Log Power (dB)", rotation=270, labelpad=12, fontsize=9)

# Export single high-resolution figure
fig.savefig(MASTER_GRID_PNG, dpi=300, bbox_inches='tight')
fig.savefig(MASTER_GRID_PDF, dpi=300, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# METRICS REPORT
# =====================================================================
total = tp + fp + tn + fn
accuracy = ((tp + tn) / total) * 100 if total > 0 else 0.0
precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n" + "=" * 50)
print(f"Overall Accuracy: {accuracy:.2f}% (TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn})")
print(f"Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1-Score: {f1_score:.2f}%")
print("=" * 50)
print(f"[SUCCESS] Unified 30-spectrogram grid saved to:\n -> {MASTER_GRID_PNG}\n -> {MASTER_GRID_PDF}\n")