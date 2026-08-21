import numpy as np
import time
import sys
sys.path.append('c:/Users/Welcome/TS_SCLF')

from rigorous_polar import (
    PolarCode, 
    encode_segmented_crc, 
    decode_ts_sclf
)

print("=========================================================")
print("      VERIFYING DYNAMIC L_SEG INTEGRITY (1000 TRIALS)")
print("=========================================================")

# Initialize both decoders
N = 128
K = 56
code_std = PolarCode(N, K)          # Default l_seg=24
code_dyn = PolarCode(N, K, l_seg=24) # Explicit l_seg=24

snr_db = 1.0
num_trials = 1000

ebno = 10**(snr_db / 10.0)
rate = 56.0 / 128.0
sigma = np.sqrt(1.0 / (2.0 * rate * ebno))

counts = {
    "Standard (Default)": {"fer": 0, "attempts": 0.0},
    "Dynamic (l_seg=24)": {"fer": 0, "attempts": 0.0}
}

np.random.seed(600) # Reintegration seed

mismatches = 0
start_time = time.time()
for trial in range(num_trials):
    payload = np.random.randint(0, 2, 48)
    
    # We use l_seg=24 for both, which is the baseline Segmented CRC
    u_I = encode_segmented_crc(payload)
    x = code_std.encode(u_I)
    
    s = 1 - 2 * x
    noise = np.random.normal(0, sigma, N)
    r = s + noise
    y = 2 * r / (sigma**2)
    
    # 1. Standard LRA-AST (Default l_seg=24)
    dec_u_I_std, success_std, attempts_std, _, _ = decode_ts_sclf(
        y, code_std, L=4, T=10, D1=16, D2=3, alpha=1.5, 
        apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
    )
    if not success_std or not np.array_equal(dec_u_I_std, u_I):
        counts["Standard (Default)"]["fer"] += 1
    counts["Standard (Default)"]["attempts"] += attempts_std
    
    # 2. Dynamic LRA-AST (Explicit l_seg=24)
    dec_u_I_dyn, success_dyn, attempts_dyn, _, _ = decode_ts_sclf(
        y, code_dyn, L=4, T=10, D1=16, D2=3, alpha=1.5, 
        apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
    )
    if not success_dyn or not np.array_equal(dec_u_I_dyn, u_I):
        counts["Dynamic (l_seg=24)"]["fer"] += 1
    counts["Dynamic (l_seg=24)"]["attempts"] += attempts_dyn
    
    # Check for mathematical identity on every trial
    if (success_std != success_dyn or 
        attempts_std != attempts_dyn or 
        not np.array_equal(dec_u_I_std, dec_u_I_dyn)):
        mismatches += 1

elapsed = time.time() - start_time
print(f"Verification completed in {elapsed:.2f} seconds.")
print("=========================================================")
print(f"Mismatches detected: {mismatches}")
print("=========================================================")

std_fer = counts["Standard (Default)"]["fer"] / num_trials
std_att = counts["Standard (Default)"]["attempts"] / num_trials

dyn_fer = counts["Dynamic (l_seg=24)"]["fer"] / num_trials
dyn_att = counts["Dynamic (l_seg=24)"]["attempts"] / num_trials

print(f"{'Decoder Scheme':<30}{'FER':<15}{'Avg. Attempts':<20}")
print("-" * 65)
print(f"{'Standard (Default l_seg)':<30}{std_fer:<15.5f}{std_att:<20.3f}")
print(f"{'Dynamic (Explicit l_seg=24)':<30}{dyn_fer:<15.5f}{dyn_att:<20.3f}")
print("=========================================================")

if mismatches == 0 and np.isclose(std_fer, dyn_fer) and np.isclose(std_att, dyn_att):
    print("VERIFICATION SUCCESSFUL: Both decoders are mathematically identical.")
else:
    print("VERIFICATION FAILED: Mismatch detected. Core logic is broken.")
print("=========================================================")
