```
PROJECT D.A.R.T. (Decentralized Acoustic Real-Time Triangulation)
IEEE SMC KGEC Summer Research Internship 2026 Deliverable
Candidate: Pratyasha Basak (Adamas University)
Mentorship: Amiya Karmakar (KGEC)


================================================================================
PROJECT OVERVIEW
================================================================================


D.A.R.T. is an edge-to-fog cybersecurity and acoustic localization architecture designed for municipal threat response and public safety. Traditional gunshot detection solutions rely on continuous streaming of raw audio to centralized cloud servers, violating citizen privacy and incurring significant continuous bandwidth costs. Furthermore, storing raw sensor claims on standard public blockchains introduces extreme storage bloat and unpredictable cryptocurrency transaction fees.

D.A.R.T. solves these dilemmas through a four-tier architecture:

Sensing Edge Layer: Captures high-frequency acoustic data into volatile circular ring buffers, extracting 128-band Log-Mel spectrograms only when an amplitude spike threshold (VRMS > 0.015) is crossed.

Ultra-Lightweight TinyML Inference: Runs an INT8-quantized 2D Convolutional Neural Network with Hybrid Global Pooling (Global Average Pooling + Global Max Pooling) locally on-device. Inference completes in under 15 milliseconds, and raw audio is immediately erased from volatile RAM to ensure 100% citizen privacy.

Zero-Gas Peer-to-Peer Integrity Validation: Detections are packaged into cryptographic headers and shared across a peer-to-peer Holochain Distributed Hash Table (DHT) network. A WebAssembly (WASM) integrity engine validates that time-of-arrival differences between nodes comply with physical acoustic propagation laws (speed of sound: 343.0 meters/second). Spoofed or simulated attacks are rejected at zero financial transaction cost.

Multilateration & Threat Prioritization: Validated acoustic timestamps are resolved through non-linear hyperbolic Time Difference of Arrival (TDoA) equations to compute real-world incident coordinates (X, Y) and calculate a Threat-Aware Priority Index (TAPI) for municipal emergency dispatch.



==============================================================================================================================================



2. SYSTEM & HARDWARE ARCHITECTURE
Physical Sensor Node (The Edge): Arduino Nano 33 BLE Sense / low-cost microcontroller mounted on municipal infrastructure. Continuously buffers 16,000 Hz single-channel audio.

Fog Gateway Node: Raspberry Pi Zero 2 W or Raspberry Pi 4 B placed inside utility enclosures, gathering BLE feature broadcasts from up to five neighboring sensor nodes.

Dual-Process Architecture:
Process A (Python Sensing Tier): Manages live audio ingestion, circular ring buffering, Mel-spectrogram feature extraction, and INT8 TFLite classification.
Process B (Holochain Ledger Tier): Maintains an immutable local source chain, communicates via local WebSockets (Port 8888), signs entries using Ed25519 keys, and runs peer validation rules over Kitsune P2P gossip networks.



======================================================================================================================================================



3. REPOSITORY DIRECTORY AND FILE STRUCTURE
DART_Team 2/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- data/
|   |-- custom_data/
|   |   |-- gunshots/ (Positive training and evaluation audio samples)
|   |   |-- negatives/ (Distractor audio: sirens, dogs, fireworks, traffic)
|   |-- unseen_test/ (Curated 30-sample real-world blind benchmark dataset)
|-- docs/
|   |-- all_30_spectrograms_grid.png (High-resolution visual spectrogram matrix)
|-- src/
|   |-- extract_features.py (Feature extraction pipeline using Librosa)
|   |-- generate_paper_grid.py (Generates the 5x6 research spectrogram grid)
|   |-- gunshot_detector.tflite (Exported INT8 quantized neural network binary)
|   |-- gunshot_listener.py (Live real-time microphone daemon with circular buffer)
|   |-- gunshot_listener_test.py (Automated batch benchmark test runner)
|   |-- mean.npy (Z-score feature normalization mean vector)
|   |-- std.npy (Z-score feature normalization standard deviation vector)
|   |-- train_model.py (2D CNN model training and INT8 quantization script)
|   |-- visualize_detections.py (Waveform and activation visualizer)
|-- zomes/
|   |-- acoustic_integrity/
|       |-- Cargo.toml (Rust compilation manifest for Holochain WebAssembly)
|       |-- src/
|           |-- lib.rs (Rust validation rules enforcing speed-of-sound physics)



=====================================================================================================================================================



4. DETAILED BREAKDOWN OF EACH FILE
src/gunshot_detector.tflite
The compiled INT8 fully quantized TensorFlow Lite model file (size: approximately 43 KB). It accepts normalized 128x63 input spectrogram tensors and produces a binary classification output (gunshot vs. non-gunshot).

src/gunshot_listener.py
The primary real-time daemon. Uses sounddevice to stream live microphone input at 16 kHz. Maintains a deque-based circular buffer, calculates Root Mean Square (RMS) energy, computes Log-Mel spectrograms upon spike detection, runs TFLite inference, and asynchronously transmits valid threat hashes over WebSockets.

src/gunshot_listener_test.py
The automated evaluation test suite. Iterates through all 30 unseen benchmark audio files in data/unseen_test/, executes windowed feature extraction, normalizes input using mean.npy and std.npy, records model confidence scores, computes inference latencies, logs deterministic Holochain transaction hashes, and outputs accuracy, precision, recall, and F1-score.

src/train_model.py
Builds the 2D Convolutional Neural Network with 2D Convolution layers, Batch Normalization, ReLU activations, SpecAugment data augmentation, and Hybrid Global Average + Max Pooling. Quantizes the trained weights from float32 down to int8 using representative datasets.

src/extract_features.py
Converts raw .wav audio into 128-band Log-Mel spectrogram arrays (Sample rate: 16000 Hz, n_fft: 512, hop_length: 256). Generates and saves mean.npy and std.npy for runtime normalization.

src/generate_paper_grid.py
Generates the 5x6 Mel-spectrogram comparison image (docs/all_30_spectrograms_grid.png) displaying positive firearm discharges side-by-side with negative environmental noise distractors.

src/visualize_detections.py
Plots detection probabilities and threshold activations over time against the input audio waveform for visual inspection.

zomes/acoustic_integrity/src/lib.rs
The Rust source file for Holochain. Contains deterministic validation callbacks enforcing the physical inequality: distance(i, j) <= (speed_of_sound * delta_t) + clock_drift.

requirements.txt
List of all required Python packages with pinned compatibility versions.

.gitignore
Excludes compiled bytecode, virtual environments, build artifacts, and oversized raw audio files (>100 MB) from version control.



============================================================================================================================================================



5. PREREQUISITES AND REQUIRED PACKAGES
Operating System: Windows 10/11, macOS, or Ubuntu Linux 20.04/22.04 LTS
Python Version: Python 3.9, 3.10, or 3.11
Hardware: Standard PC/Laptop with working microphone input, or Raspberry Pi / embedded Linux SBC.

Required Python Libraries:

numpy: Tensor manipulation and mathematical computations

scipy: Multilateration optimization and digital signal processing

librosa: Audio feature transformation, Log-Mel spectrogram generation

soundfile: Audio I/O backend for wav decoding

sounddevice: Low-latency real-time microphone stream capture

tflite-runtime or tensorflow: Neural network execution engine

matplotlib: Spectrogram rendering and metric plotting

scikit-learn: Confusion matrix calculation and performance metrics

websockets: Local inter-process communication with Holochain conductor


====================================================================================================================================================



6. STEP-BY-STEP SETUP AND INSTALLATION GUIDE
Step 1: Clone the Official Repository
Open a terminal (PowerShell, Command Prompt, or Bash) and run:
git clone https://github.com/ieeesmckgec-student-branch-chapter/IEEE-SMC-KGEC-Internship-2026.git
cd IEEE-SMC-KGEC-Internship-2026
git checkout internships
cd "DART_Team 2"

Step 2: Create and Activate a Python Virtual Environment
For Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1

For Linux / macOS:
python3 -m venv venv
source venv/bin/activate

Step 3: Install Required Dependencies
pip install --upgrade pip
pip install -r requirements.txt

If you encounter issues installing sounddevice or librosa, install system build libraries:
On Linux (Ubuntu/Debian):
sudo apt-get update
sudo apt-get install libportaudio2 libasound-dev libsndfile1



====================================================================================================================================================



7. RUNNING THE SYSTEM
Option A: Run the 30-Sample Benchmark Evaluation Suite
To verify the full detection and simulated Holochain hash dispatch pipeline on the test benchmark dataset, run:
python src/gunshot_listener_test.py

Expected output:

Sequential evaluation of 30 unseen audio samples (firearms, fireworks, sirens, crowd noise).

Real-time classification confidence scores, inference latency (under 15 ms/window), and generated transaction hashes.

Final summary matrix: Accuracy: 76.67%, Precision: 71.43%, Recall: 76.92%, F1-Score: 74.07%.

Option B: Run the Live Real-Time Microphone Listener
To monitor physical microphone input in real-time:
python src/gunshot_listener.py

Behavior:

Continuously listens in the background using minimal CPU.

When an acoustic spike exceeds the RMS energy threshold, it extracts the spectrogram, runs the INT8 model, and logs a detection alert if verified.

Immediately wipes raw buffer data from memory.

Option C: Generate Spectrogram Comparison Grids
To reconstruct the research publication image:
python src/generate_paper_grid.py

The output grid will be generated and saved to docs/all_30_spectrograms_grid.png.



============================================================================================================================================================



8. TROUBLESHOOTING AND ISSUE RESOLUTION GUIDE
Issue 1: SoundDevice PortAudio Error / No Default Input Device Found
Symptom: sounddevice.PortAudioError: Error opening InputStream: Invalid number of channels / No default input device.
Resolution:

Ensure your physical microphone is plugged in, recognized by Windows/Linux, and enabled in privacy settings.

In Python, check available audio devices:
python -c "import sounddevice; print(sounddevice.query_devices())"

In gunshot_listener.py, explicitly pass the device index:
sd.InputStream(device=1, channels=1, samplerate=16000, ...)

Issue 2: GitHub Rejection Due to Large Files (>100 MB Limit)
Symptom: remote error: File exceeds GitHub's file size limit of 100.00 MB.
Resolution:

GitHub blocks commits containing single files over 100 MB (such as raw high-duration audio compilations).

Undo the local commit and remove the file from git tracking without deleting it from disk:
git reset --soft HEAD~1
git rm --cached "path/to/large_file.wav"
git commit -m "Submit clean project files"
git push -u origin internships

Issue 3: Librosa PySoundFile or Numba Compatibility Warning
Symptom: UserWarning: PySoundFile failed. Trying audioread instead.
Resolution:

Ensure soundfile and resampy are installed:
pip install soundfile resampy

Librosa requires audio buffers in 32-bit floating point format normalized between -1.0 and 1.0. The codebase automatically handles float32 casting during ring buffer ingestion.

Issue 4: WebSocket Connection Refused (Port 8888)
Symptom: ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8888).
Resolution:

This indicates the Python sensing daemon is attempting to dispatch a detection to the local Holochain conductor when the conductor is not running in the background.

For standalone testing, gunshot_listener_test.py includes a fallback mock hash generator that computes Ed25519-compatible payload hashes deterministically even if the local Holochain conductor daemon is offline.


====================================================================================================================================================


9. EXPERIMENTAL EVALUATION & RESULTS
Evaluated on 30 unseen blind test scenarios consisting of firearm discharges (rifles, shotguns, handguns, automatic bursts) and challenging urban distractors (firework shells, sirens, dog barking, traffic, crowd noise):

Total Samples Tested: 30

True Positives (TP): 10 (Firearms correctly detected)

True Negatives (TN): 13 (Distractors correctly rejected)

False Positives (FP): 4 (High-energy sharp pyrotechnics)

False Negatives (FN): 3 (Heavily suppressed/distant discharges)

Classification Accuracy: 76.67%

Model Precision: 71.43%

Model Recall: 76.92%

Model F1-Score: 74.07%

Average Edge Inference Latency: 11.4 ms (on ARM Cortex-A53 / x86_64)

Flash / RAM Footprint: ~43 KB (Model), <2.5 MB peak execution RAM


================================================================================================================


10. LICENSE & ACKNOWLEDGMENTS
This project was developed as an official deliverable for the IEEE SMC KGEC Summer Research Internship 2026 under the mentorship of Amiya Karmakar (KGEC).

All code and architecture are released under the MIT License.

```