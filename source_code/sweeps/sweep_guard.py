import numpy as np
import time
from rigorous_polar import (
    PolarCode, 
    encode_segmented_crc, 
    decode_ca_scl, 
    decode_sclf, 
    decode_ts_sclf
)

print("=========================================================")
# Initialize Codec
N = 128
K = 56
code = PolarCode(N, K)

snr_db = 1.0
num_trials = 100

print(f"Running guard-band sweep test: {num_trials} trials at {snr_db} dB")
print("=========================================================")

ebno = 10**(snr_db / 10.0)
rate = 56.0 / 128.0
sigma = np.sqrt(1.0 / (2.0 * rate * ebno))

# We will sweep these configurations
configs = [
    ("CA-SCL", lambda y: decode_ca_scl(y, code, L=4)),
    ("SCLF", lambda y: decode_sclf(y, code, L=4, T=10, D1=16, D2=3)),
    ("TS-SCLF (Constraint, G=4)", lambda y: decode_ts_sclf(y, code, L=4, T=10, D1=16, D2=3, alpha=0.5, apply_constraint=True, guard_band=4)),
    ("TS-SCLF (Constraint, G=8)", lambda y: decode_ts_sclf(y, code, L=4, T=10, D1=16, D2=3, alpha=0.5, apply_constraint=True, guard_band=8)),
    ("TS-SCLF (Constraint, G=12)", lambda y: decode_ts_sclf(y, code, L=4, T=10, D1=16, D2=3, alpha=0.5, apply_constraint=True, guard_band=12)),
    ("TS-SCLF (Constraint, G=16)", lambda y: decode_ts_sclf(y, code, L=4, T=10, D1=16, D2=3, alpha=0.5, apply_constraint=True, guard_band=16)),
    ("TS-SCLF (No Constraint, G=12)", lambda y: decode_ts_sclf(y, code, L=4, T=10, D1=16, D2=3, alpha=0.5, apply_constraint=False, guard_band=12))
]

counts = {name: {"fer": 0, "attempts": 0.0} for name, _ in configs}

start_time = time.time()
for trial in range(num_trials):
    # Generate random payload and encode
    payload = np.random.randint(0, 2, 48)
    u_I = encode_segmented_crc(payload)
    x = code.encode(u_I)
    
    # Modulation + AWGN Channel
    s = 1 - 2 * x
    noise = np.random.normal(0, sigma, N)
    r = s + noise
    y = 2 * r / (sigma**2)
    
    # Run each config
    for name, decode_fn in configs:
        res = decode_fn(y)
        dec_u_I = res[0]
        success = res[1]
        attempts = res[2] if len(res) >= 3 else 1
        
        if not success or not np.array_equal(dec_u_I, u_I):
            counts[name]["fer"] += 1
        counts[name]["attempts"] += attempts

elapsed = time.time() - start_time
print(f"Sweep completed in {elapsed:.2f} seconds.")
print("=========================================================================================")
print(f"{'Scheme':<35}{'FER':<15}{'Avg. Attempts':<20}{'Attempts vs SCLF':<20}")
print("-----------------------------------------------------------------------------------------")
sclf_attempts = counts["SCLF"]["attempts"] / num_trials
for name, _ in configs:
    fer = counts[name]["fer"] / num_trials
    attempts = counts[name]["attempts"] / num_trials
    reduction = ((sclf_attempts - attempts) / sclf_attempts * 100.0) if "SCLF" not in name and "CA-SCL" not in name else 0.0
    reduction_str = f"{reduction:+.1f}%" if "SCLF" not in name and "CA-SCL" not in name else "-"
    print(f"{name:<35}{fer:<15.5f}{attempts:<20.3f}{reduction_str:<20}")
print("=========================================================================================")
