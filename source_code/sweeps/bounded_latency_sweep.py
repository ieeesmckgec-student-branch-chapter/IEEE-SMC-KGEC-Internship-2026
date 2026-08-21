import numpy as np
import time
import sys
sys.path.append('c:/Users/Welcome/TS_SCLF')

from rigorous_polar import (
    PolarCode, 
    encode_segmented_crc, 
    decode_sclf, 
    decode_ts_sclf,
    decode_bounded_lra_ast
)

print("=========================================================")
print("      GAP 4: BOUNDED LATENCY COMPARATIVE SWEEP (0.5 dB)")
print("=========================================================")

# Initialize Codecs
N = 128
K = 56
code = PolarCode(N, K)

snr_db = 0.5
num_trials = 1000

print(f"Running 1000 trials at {snr_db} dB synchronously...")
print("=========================================================")

ebno = 10**(snr_db / 10.0)
rate = 56.0 / 128.0
sigma = np.sqrt(1.0 / (2.0 * rate * ebno))

# Histograms will store count for attempts 1, 2, 3, ... up to 11
histograms = {
    "SCLF (T=10)": np.zeros(12, dtype=int),
    "LRA-AST (T=10)": np.zeros(12, dtype=int),
    "Bounded LRA-AST (T=6)": np.zeros(12, dtype=int)
}

counts = {
    "SCLF (T=10)": {"fer": 0, "attempts": 0.0, "worst_case": 0},
    "LRA-AST (T=10)": {"fer": 0, "attempts": 0.0, "worst_case": 0},
    "Bounded LRA-AST (T=6)": {"fer": 0, "attempts": 0.0, "worst_case": 0}
}

np.random.seed(400) # Simulation seed

start_time = time.time()
for trial in range(num_trials):
    payload = np.random.randint(0, 2, 48)
    u_I = encode_segmented_crc(payload)
    x = code.encode(u_I)
    
    s = 1 - 2 * x
    noise = np.random.normal(0, sigma, N)
    r = s + noise
    y_channel = 2 * r / (sigma**2)
    
    # 1. SCLF (T=10)
    dec_u_I_sclf, success_sclf, attempts_sclf = decode_sclf(y_channel, code, L=4, T=10, D1=16, D2=3)
    if not success_sclf or not np.array_equal(dec_u_I_sclf, u_I):
        counts["SCLF (T=10)"]["fer"] += 1
    counts["SCLF (T=10)"]["attempts"] += attempts_sclf
    counts["SCLF (T=10)"]["worst_case"] = max(counts["SCLF (T=10)"]["worst_case"], attempts_sclf)
    histograms["SCLF (T=10)"][attempts_sclf] += 1
    
    # 2. LRA-AST (T=10)
    dec_u_I_lra, success_lra, attempts_lra, _, _ = decode_ts_sclf(
        y_channel, code, L=4, T=10, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
    )
    if not success_lra or not np.array_equal(dec_u_I_lra, u_I):
        counts["LRA-AST (T=10)"]["fer"] += 1
    counts["LRA-AST (T=10)"]["attempts"] += attempts_lra
    counts["LRA-AST (T=10)"]["worst_case"] = max(counts["LRA-AST (T=10)"]["worst_case"], attempts_lra)
    histograms["LRA-AST (T=10)"][attempts_lra] += 1
    
    # 3. Bounded LRA-AST (T=6)
    dec_u_I_bounded, success_bounded, attempts_bounded, _, _ = decode_bounded_lra_ast(
        y_channel, code, L=4, T=6, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
    )
    if not success_bounded or not np.array_equal(dec_u_I_bounded, u_I):
        counts["Bounded LRA-AST (T=6)"]["fer"] += 1
    counts["Bounded LRA-AST (T=6)"]["attempts"] += attempts_bounded
    counts["Bounded LRA-AST (T=6)"]["worst_case"] = max(counts["Bounded LRA-AST (T=6)"]["worst_case"], attempts_bounded)
    histograms["Bounded LRA-AST (T=6)"][attempts_bounded] += 1

elapsed = time.time() - start_time
print(f"Sweep completed in {elapsed:.2f} seconds.")
print("=========================================================")

# Print Summary Table
print(f"{'Decoder Scheme':<25}{'FER':<12}{'Avg. Attempts':<18}{'Worst-Case Attempts':<20}")
print("-" * 75)
for scheme in counts.keys():
    fer = counts[scheme]["fer"] / num_trials
    attempts = counts[scheme]["attempts"] / num_trials
    worst = counts[scheme]["worst_case"]
    print(f"{scheme:<25}{fer:<12.5f}{attempts:<18.3f}{worst:<20}")
print("=========================================================")

# Print Histograms
print("\nATTEMPTS HISTOGRAMS (Frame Counts):")
print("-" * 75)
print(f"{'Attempts':<10}{'SCLF (T=10)':<20}{'LRA-AST (T=10)':<20}{'Bounded LRA-AST (T=6)':<20}")
print("-" * 75)
for a in range(1, 12):
    sclf_cnt = histograms["SCLF (T=10)"][a]
    lra_cnt = histograms["LRA-AST (T=10)"][a]
    bounded_cnt = histograms["Bounded LRA-AST (T=6)"][a]
    print(f"{a:<10}{sclf_cnt:<20}{lra_cnt:<20}{bounded_cnt:<20}")
print("=========================================================")
