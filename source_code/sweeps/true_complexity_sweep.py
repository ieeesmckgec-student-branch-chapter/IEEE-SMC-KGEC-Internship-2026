import numpy as np
import time
from rigorous_polar import (
    PolarCode, 
    encode_segmented_crc, 
    decode_ca_scl, 
    decode_sclf, 
    decode_ts_sclf
)

# We will patch the update_paths_at_leaf in rigorous_polar to track the number of decoded bits.
# To do this cleanly without modifying the file again, we can simply track the bits decoded:
# In SCL tree decoding, the decoder processes layers l from 0 to N-1.
# If a run completes, it decodes N bits.
# If a run is terminated early (paths list becomes empty) at layer l, it decodes l bits.
# So we can easily track this by returning the termination layer l from the decoder,
# or by counting how many layers were processed!
# Let's write a modified simulation loop that inspects this.
# Wait! In rigorous_polar.py, when a run is terminated:
# - In Segmented CRC, it terminates at l = l_seg1 (which is index 71 for info index 26).
# - In SCL-RE early termination, it terminates at some information layer l.
# Let's count the exact number of decoded bits by tracking the termination point!

class ComplexityTracker:
    def __init__(self):
        self.total_bits = 0

print("=========================================================")
print("      TRUE COMPUTATIONAL COMPLEXITY SWEEP (N=128)")
print("=========================================================")

# Initialize Codecs
N = 128
K = 56
code_ca = PolarCode(N, K)
code_sclf = PolarCode(N, K)
code_ts = PolarCode(N, K)
code_lra = PolarCode(N, K)

snr_db = 1.0
num_trials = 500

print(f"Running 500-trial true complexity sweep at {snr_db} dB")
print("=========================================================")

ebno = 10**(snr_db / 10.0)
rate = 56.0 / 128.0
sigma = np.sqrt(1.0 / (2.0 * rate * ebno))

# To track true complexity, we will count:
# - CA-SCL: always 1 run of N bits = 128 bits.
# - SCLF: 1 initial run (128 bits) + R restarts (each 128 bits). Total = (1 + R) * 128 bits.
# - TS-SCLF / LRA-AST: 
#   - Initial SCL run: 
#     - If succeeds: 128 bits.
#     - If fails Segment 1 CRC: terminated at l_seg1.
#       Wait, what is l_seg1? It is code.l_seg1 (codeword index).
#       The number of leaf bits decoded before l_seg1 is l_seg1 + 1.
#   - Restarted runs:
#     - If succeeds: 128 bits.
#     - If terminated by Segment 1 CRC: l_seg1 + 1 bits.
#     - If terminated by SCL-RE early termination: l + 1 bits.
#
# Let's instrument the decode functions to return the exact number of decoded bits!
# We can do this by wrapping update_paths_at_leaf to count how many times it is called.
# Since update_paths_at_leaf is called once per leaf, the number of times it is called
# is exactly the number of decoded leaf bits!
# This is mathematically perfect and extremely easy to instrument!

import rigorous_polar

# We will monkey-patch rigorous_polar.update_paths_at_leaf to count calls
original_update = rigorous_polar.update_paths_at_leaf

active_tracker = None

def patched_update(l, paths, code, L, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer, guard_band=12, use_lra_ast=False, gamma_lra=0.8):
    if active_tracker is not None:
        active_tracker.total_bits += 1
    return original_update(l, paths, code, L, is_first_round, first_round_pms, first_round_llrs, alpha, is_flipping, flip_layer, guard_band, use_lra_ast, gamma_lra)

rigorous_polar.update_paths_at_leaf = patched_update

schemes = ["CA-SCL", "SCLF", "TS-SCLF (Global alpha=1.0)", "LRA-AST (Adaptive alpha)"]
counts = {k: {"fer": 0, "complexity": 0.0, "attempts": 0.0} for k in schemes}

tracker = ComplexityTracker()

for trial in range(num_trials):
    payload = np.random.randint(0, 2, 48)
    u_I = encode_segmented_crc(payload)
    x = code_ca.encode(u_I)
    
    s = 1 - 2 * x
    noise = np.random.normal(0, sigma, N)
    r = s + noise
    y = 2 * r / (sigma**2)
    
    # 1. CA-SCL
    active_tracker = tracker
    tracker.total_bits = 0
    dec_u_I_scl, success_scl = decode_ca_scl(y, code_ca, L=4)
    if not success_scl or not np.array_equal(dec_u_I_scl, u_I):
        counts["CA-SCL"]["fer"] += 1
    counts["CA-SCL"]["complexity"] += tracker.total_bits / N
    counts["CA-SCL"]["attempts"] += 1
    
    # 2. SCLF
    tracker.total_bits = 0
    dec_u_I_sclf, success_sclf, attempts_sclf = decode_sclf(y, code_sclf, L=4, T=10, D1=16, D2=3)
    if not success_sclf or not np.array_equal(dec_u_I_sclf, u_I):
        counts["SCLF"]["fer"] += 1
    counts["SCLF"]["complexity"] += tracker.total_bits / N
    counts["SCLF"]["attempts"] += attempts_sclf
    
    # 3. TS-SCLF (Global alpha=1.0, G=8)
    tracker.total_bits = 0
    dec_u_I_ts, success_ts, attempts_ts, _, _ = decode_ts_sclf(
        y, code_ts, L=4, T=10, D1=16, D2=3, alpha=1.0, apply_constraint=True, guard_band=8, use_lra_ast=False
    )
    if not success_ts or not np.array_equal(dec_u_I_ts, u_I):
        counts["TS-SCLF (Global alpha=1.0)"]["fer"] += 1
    counts["TS-SCLF (Global alpha=1.0)"]["complexity"] += tracker.total_bits / N
    counts["TS-SCLF (Global alpha=1.0)"]["attempts"] += attempts_ts
    
    # 4. LRA-AST (Adaptive alpha, alpha_0=1.5, gamma=0.5, G=8)
    tracker.total_bits = 0
    dec_u_I_lra, success_lra, attempts_lra, _, _ = decode_ts_sclf(
        y, code_lra, L=4, T=10, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
    )
    if not success_lra or not np.array_equal(dec_u_I_lra, u_I):
        counts["LRA-AST (Adaptive alpha)"]["fer"] += 1
    counts["LRA-AST (Adaptive alpha)"]["complexity"] += tracker.total_bits / N
    counts["LRA-AST (Adaptive alpha)"]["attempts"] += attempts_lra

active_tracker = None
rigorous_polar.update_paths_at_leaf = original_update

print("=========================================================================================")
print(f"{'Scheme':<30}{'FER':<12}{'Avg. Attempts':<18}{'True Complexity':<20}{'Complexity Reduction':<20}")
print("-----------------------------------------------------------------------------------------")
sclf_complexity = counts["SCLF"]["complexity"] / num_trials
for scheme in schemes:
    fer = counts[scheme]["fer"] / num_trials
    attempts = counts[scheme]["attempts"] / num_trials
    complexity = counts[scheme]["complexity"] / num_trials
    reduction = ((sclf_complexity - complexity) / sclf_complexity * 100.0) if scheme in ["TS-SCLF (Global alpha=1.0)", "LRA-AST (Adaptive alpha)"] else 0.0
    reduction_str = f"{reduction:+.2f}%" if scheme in ["TS-SCLF (Global alpha=1.0)", "LRA-AST (Adaptive alpha)"] else "-"
    print(f"{scheme:<30}{fer:<12.5f}{attempts:<18.3f}{complexity:<20.3f}{reduction_str:<20}")
print("=========================================================================================")
