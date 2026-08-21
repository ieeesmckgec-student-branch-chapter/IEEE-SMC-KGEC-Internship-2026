import numpy as np
import time
import sys
sys.path.append('c:/Users/Welcome/TS_SCLF')

from rigorous_polar import (
    PolarCode, 
    encode_segmented_crc, 
    decode_sclf, 
    decode_ts_sclf,
    Path,
    decode_node,
    check_crc,
    POLY_CRC5,
    run_flipskip
)

# =====================================================================
# 1. NumPy Logistic Regression Implementation
# =====================================================================
class NumPyLogisticRegression:
    def __init__(self, lr=0.1, epochs=1500):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.mean = None
        self.std = None
        
    def fit(self, X, y):
        # Standardize features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - self.mean) / self.std
        
        num_samples, num_features = X_scaled.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        
        # Gradient descent with L2 regularization
        lambda_reg = 0.1
        for _ in range(self.epochs):
            linear_model = np.dot(X_scaled, self.weights) + self.bias
            y_predicted = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -15, 15)))
            
            # Gradients
            dw = (1.0 / num_samples) * np.dot(X_scaled.T, (y_predicted - y)) + (lambda_reg / num_samples) * self.weights
            db = (1.0 / num_samples) * np.sum(y_predicted - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_proba(self, X):
        X_scaled = (X - self.mean) / self.std
        linear_model = np.dot(X_scaled, self.weights) + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(linear_model, -15, 15)))

# =====================================================================
# 2. Calibration Phase: Collecting Training Data with Reached-Layer Constraint
# =====================================================================
print("=========================================================")
print("      GAP 2: LEARNED FLIPPING SET ORDERING CALIBRATION")
print("=========================================================")
N = 128
K = 56
code = PolarCode(N, K)
snr_db = 1.0
ebno = 10**(snr_db / 10.0)
rate = 56.0 / 128.0
sigma = np.sqrt(1.0 / (2.0 * rate * ebno))

num_calib_frames = 300
print(f"Collecting training data from {num_calib_frames} calibration frames at 1.0 dB...")

X_train_list = []
y_train_list = []

np.random.seed(200) # Calibration seed

failed_calib_count = 0
for frame in range(num_calib_frames):
    payload = np.random.randint(0, 2, 48)
    u_I = encode_segmented_crc(payload)
    x = code.encode(u_I)
    
    s = 1 - 2 * x
    noise = np.random.normal(0, sigma, N)
    r = s + noise
    y_channel = 2 * r / (sigma**2)
    
    # Run first round SCL (L=4)
    first_path = Path(code.n, code.N)
    first_path.llrs[0, :] = y_channel
    paths = [first_path]
    
    first_round_llrs = {}
    decode_node(
        0, 0, paths, code, L=4, 
        is_first_round=True, 
        first_round_pms=None, 
        first_round_llrs=first_round_llrs, 
        alpha=1.5, 
        is_flipping=False, 
        flip_layer=-1, 
        guard_band=8, 
        use_lra_ast=True, 
        gamma_lra=0.5
    )
    
    succeeded = False
    for path in paths:
        u_I_dec = path.bits[code.n, code.info_indices]
        if check_crc(u_I_dec, POLY_CRC5):
            succeeded = True
            break
            
    if not succeeded:
        failed_calib_count += 1
        # Only collect training features for layers that were ACTUALLY reached in the first round!
        for l in code.info_indices:
            if l in first_round_llrs:
                # Features: LLR magnitude, polarization weight, and codeword index
                llr_val = abs(first_round_llrs[l])
                pw = code.weights[l]
                idx = l
                
                # Label: 1 if flipping this layer corrects the frame, 0 otherwise
                restart_path = Path(code.n, code.N)
                restart_path.llrs[0, :] = y_channel
                restart_paths = [restart_path]
                
                decode_node(
                    0, 0, restart_paths, code, L=4,
                    is_first_round=False, first_round_pms=None, first_round_llrs=None,
                    alpha=1.5, is_flipping=True, flip_layer=l, guard_band=8,
                    use_lra_ast=True, gamma_lra=0.5
                )
                
                corrected = False
                for path in restart_paths:
                    u_I_dec = path.bits[code.n, code.info_indices]
                    if check_crc(u_I_dec, POLY_CRC5):
                        corrected = True
                        break
                
                X_train_list.append([llr_val, pw, idx])
                y_train_list.append(1.0 if corrected else 0.0)

print(f"Calibration complete. Failed frames: {failed_calib_count} / {num_calib_frames}")
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

# Train Model
model = NumPyLogisticRegression()
if len(X_train) > 0:
    model.fit(X_train, y_train)
    print("Logistic regression model trained successfully.")
    print(f"Learned coefficients (LLR magnitude, PW, index): {model.weights}")
    print(f"Learned bias: {model.bias}")
else:
    print("No failed frames in calibration. Using fallback model.")
    model.weights = np.array([-1.0, 0.0, 0.0]) # Fallback

# =====================================================================
# 3. Implementing the Learned Flipping Set Decoder
# =====================================================================
def decode_learned_flip_sclf(y, code, model, L=4, T=10, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5):
    # --- Round 0: Initial SCL Decoding ---
    first_path = Path(code.n, code.N)
    first_path.llrs[0, :] = y
    paths = [first_path]
    
    first_round_pms = {}
    first_round_llrs = {}
    
    decode_node(0, 0, paths, code, L, is_first_round=True, first_round_pms=first_round_pms, 
                first_round_llrs=first_round_llrs, alpha=alpha, is_flipping=False, flip_layer=-1, guard_band=guard_band, use_lra_ast=use_lra_ast, gamma_lra=gamma_lra)
    
    # Check if initial SCL decoding succeeded
    for path in paths:
        u_I = path.bits[code.n, code.info_indices]
        if check_crc(u_I, POLY_CRC5):
            return u_I, True, 1
            
    failed_seg1 = (len(paths) == 0)
    
    # Generate flipping set candidate layers
    info_layers = list(code.info_indices)
    if failed_seg1 and apply_constraint:
        info_layers = [l for l in info_layers if l <= code.l_seg1]
        
    # Extract features for reached candidate layers
    reached_layers = []
    features = []
    for l in info_layers:
        if l in first_round_llrs:
            llr_val = abs(first_round_llrs[l])
            pw = code.weights[l]
            idx = l
            features.append([llr_val, pw, idx])
            reached_layers.append(l)
            
    if len(features) > 0:
        # Predict success probabilities using our trained model
        probs = model.predict_proba(np.array(features))
        
        # Sort reached layers by predicted probability in descending order
        sorted_indices = np.argsort(probs)[::-1]
        S = [reached_layers[idx] for idx in sorted_indices]
    else:
        # Fallback to standard sorting if no layers were reached (should not happen)
        S = sorted(info_layers, key=lambda l: abs(first_round_llrs.get(l, 999.0)))
    
    attempts = 1
    # --- Flipping Loop ---
    for i in range(min(T, len(S))):
        # Check FLIPSKIP
        fout = run_flipskip(S, i, D1, D2)
        if fout == 1:
            continue
            
        attempts += 1
            
        flip_layer = S[i]
        restart_path = Path(code.n, code.N)
        restart_path.llrs[0, :] = y
        restart_paths = [restart_path]
        
        decode_node(0, 0, restart_paths, code, L, is_first_round=False, first_round_pms=first_round_pms,
                    first_round_llrs=None, alpha=alpha, is_flipping=True, flip_layer=flip_layer, guard_band=guard_band, use_lra_ast=use_lra_ast, gamma_lra=gamma_lra)
        
        # Check if any path passed final CRC
        for path in restart_paths:
            u_I = path.bits[code.n, code.info_indices]
            if check_crc(u_I, POLY_CRC5):
                return u_I, True, attempts
                
    if len(paths) > 0:
        return paths[0].bits[code.n, code.info_indices], False, attempts
    return np.zeros(code.K, dtype=int), False, attempts

# =====================================================================
# 4. Comparative Simulation Sweep (1,000 Trials)
# =====================================================================
print("\n=========================================================")
print("      GAP 2: 1000-TRIAL COMPARATIVE SIMULATION SWEEP")
print("=========================================================")
num_trials = 1000
print(f"Running 1000 trials at {snr_db} dB synchronously...")

counts = {
    "SCLF (L=4)": {"fer": 0, "attempts": 0.0},
    "Learned SCLF": {"fer": 0, "attempts": 0.0}
}

np.random.seed(300) # Simulation seed for comparison

start_time = time.time()
for trial in range(num_trials):
    payload = np.random.randint(0, 2, 48)
    u_I = encode_segmented_crc(payload)
    x = code.encode(u_I)
    
    s = 1 - 2 * x
    noise = np.random.normal(0, sigma, N)
    r = s + noise
    y_channel = 2 * r / (sigma**2)
    
    # 1. Standard SCLF (L=4)
    dec_u_I_sclf, success_sclf, attempts_sclf = decode_sclf(y_channel, code, L=4, T=10, D1=16, D2=3)
    if not success_sclf or not np.array_equal(dec_u_I_sclf, u_I):
        counts["SCLF (L=4)"]["fer"] += 1
    counts["SCLF (L=4)"]["attempts"] += attempts_sclf
    
    # 2. Learned SCLF
    dec_u_I_learned, success_learned, attempts_learned = decode_learned_flip_sclf(
        y_channel, code, model, L=4, T=10, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
    )
    if not success_learned or not np.array_equal(dec_u_I_learned, u_I):
        counts["Learned SCLF"]["fer"] += 1
    counts["Learned SCLF"]["attempts"] += attempts_learned

elapsed = time.time() - start_time
print(f"Sweep completed in {elapsed:.2f} seconds.")
print("=========================================================")

sclf_fer = counts["SCLF (L=4)"]["fer"] / num_trials
sclf_att = counts["SCLF (L=4)"]["attempts"] / num_trials

learned_fer = counts["Learned SCLF"]["fer"] / num_trials
learned_att = counts["Learned SCLF"]["attempts"] / num_trials

print(f"{'Decoder Scheme':<25}{'FER':<15}{'Avg. Attempts':<20}")
print("-" * 60)
print(f"{'SCLF (L=4) baseline':<25}{sclf_fer:<15.5f}{sclf_att:<20.3f}")
print(f"{'Learned SCLF (Proposed)':<25}{learned_fer:<15.5f}{learned_att:<20.3f}")
print("=========================================================")
