import numpy as np
import time
from rigorous_polar import (
    PolarCode, 
    encode_segmented_crc, 
    decode_ca_scl, 
    decode_sclf, 
    decode_ts_sclf
)

print("=================================================================")
print("      TS-SCLF VS LRA-AST FULL Monte Carlo SNR SWEEP")
print("=================================================================")

# Initialize Codecs
N = 128
K = 56
code_ca = PolarCode(N, K)
code_sclf = PolarCode(N, K)
code_ts = PolarCode(N, K)
code_lra = PolarCode(N, K)

snr_db_list = [0.5, 1.0, 1.5, 2.0]
num_trials = 1000

results = {
    "CA-SCL": {"fer": [], "attempts": []},
    "SCLF": {"fer": [], "attempts": []},
    "TS-SCLF": {"fer": [], "attempts": [], "fires": []},
    "LRA-AST": {"fer": [], "attempts": [], "fires": []}
}

total_start_time = time.time()

for snr_db in snr_db_list:
    print(f"\nSimulating SNR = {snr_db:.1f} dB...")
    
    # Reset early termination counters for this SNR point
    code_ts.early_termination_count = 0
    code_lra.early_termination_count = 0
    
    ebno = 10**(snr_db / 10.0)
    rate = 56.0 / 128.0
    sigma = np.sqrt(1.0 / (2.0 * rate * ebno))
    
    counts = {
        "CA-SCL": {"fer": 0, "attempts": 0.0},
        "SCLF": {"fer": 0, "attempts": 0.0},
        "TS-SCLF": {"fer": 0, "attempts": 0.0},
        "LRA-AST": {"fer": 0, "attempts": 0.0}
    }
    
    start_time = time.time()
    for trial in range(num_trials):
        # Generate payload and encode
        payload = np.random.randint(0, 2, 48)
        u_I = encode_segmented_crc(payload)
        x = code_ca.encode(u_I)
        
        # BPSK Modulation + AWGN Channel
        s = 1 - 2 * x
        noise = np.random.normal(0, sigma, N)
        r = s + noise
        y = 2 * r / (sigma**2)
        
        # 1. CA-SCL
        dec_u_I_scl, success_scl = decode_ca_scl(y, code_ca, L=4)
        if not success_scl or not np.array_equal(dec_u_I_scl, u_I):
            counts["CA-SCL"]["fer"] += 1
        counts["CA-SCL"]["attempts"] += 1
        
        # 2. SCLF
        dec_u_I_sclf, success_sclf, attempts_sclf = decode_sclf(y, code_sclf, L=4, T=10, D1=16, D2=3)
        if not success_sclf or not np.array_equal(dec_u_I_sclf, u_I):
            counts["SCLF"]["fer"] += 1
        counts["SCLF"]["attempts"] += attempts_sclf
        
        # 3. TS-SCLF (Global alpha=1.0, G=8)
        dec_u_I_ts, success_ts, attempts_ts, _, _ = decode_ts_sclf(
            y, code_ts, L=4, T=10, D1=16, D2=3, alpha=1.0, apply_constraint=True, guard_band=8, use_lra_ast=False
        )
        if not success_ts or not np.array_equal(dec_u_I_ts, u_I):
            counts["TS-SCLF"]["fer"] += 1
        counts["TS-SCLF"]["attempts"] += attempts_ts
        
        # 4. LRA-AST (Adaptive alpha, alpha_0=1.5, gamma=0.5, G=8)
        dec_u_I_lra, success_lra, attempts_lra, _, _ = decode_ts_sclf(
            y, code_lra, L=4, T=10, D1=16, D2=3, alpha=1.5, apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
        )
        if not success_lra or not np.array_equal(dec_u_I_lra, u_I):
            counts["LRA-AST"]["fer"] += 1
        counts["LRA-AST"]["attempts"] += attempts_lra

    elapsed = time.time() - start_time
    print(f"SNR {snr_db:.1f} dB completed in {elapsed:.2f} seconds.")
    
    # Store results
    results["CA-SCL"]["fer"].append(counts["CA-SCL"]["fer"] / num_trials)
    results["CA-SCL"]["attempts"].append(counts["CA-SCL"]["attempts"] / num_trials)
    
    results["SCLF"]["fer"].append(counts["SCLF"]["fer"] / num_trials)
    results["SCLF"]["attempts"].append(counts["SCLF"]["attempts"] / num_trials)
    
    results["TS-SCLF"]["fer"].append(counts["TS-SCLF"]["fer"] / num_trials)
    results["TS-SCLF"]["attempts"].append(counts["TS-SCLF"]["attempts"] / num_trials)
    results["TS-SCLF"]["fires"].append(code_ts.early_termination_count)
    
    results["LRA-AST"]["fer"].append(counts["LRA-AST"]["fer"] / num_trials)
    results["LRA-AST"]["attempts"].append(counts["LRA-AST"]["attempts"] / num_trials)
    results["LRA-AST"]["fires"].append(code_lra.early_termination_count)

total_elapsed = time.time() - total_start_time
print(f"\nAll simulations completed in {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes).")

# Print beautiful comparative summary table
print("\n" + "="*115)
print(f"{'SNR (dB)':<10}{'Scheme':<30}{'FER':<15}{'Avg. Attempts':<20}{'Complexity Reduction vs SCLF':<30}")
print("="*115)
for idx, snr_db in enumerate(snr_db_list):
    sclf_attempts = results["SCLF"]["attempts"][idx]
    for scheme in ["CA-SCL", "SCLF", "TS-SCLF", "LRA-AST"]:
        fer = results[scheme]["fer"][idx]
        attempts = results[scheme]["attempts"][idx]
        reduction = ((sclf_attempts - attempts) / sclf_attempts * 100.0) if scheme in ["TS-SCLF", "LRA-AST"] else 0.0
        reduction_str = f"{reduction:+.2f}%" if scheme in ["TS-SCLF", "LRA-AST"] else "-"
        print(f"{snr_db:<10.1f}{scheme:<30}{fer:<15.5f}{attempts:<20.3f}{reduction_str:<30}")
    print("-"*115)

# Print early termination fire counts table
print("\n" + "="*65)
print(f"{'SNR (dB)':<10}{'Scheme':<30}{'Early Terminations Fired':<25}")
print("="*65)
for idx, snr_db in enumerate(snr_db_list):
    print(f"{snr_db:<10.1f}{'TS-SCLF (Global alpha=1.0)':<30}{results['TS-SCLF']['fires'][idx]:<25}")
    print(f"{'':<10}{'LRA-AST (Adaptive alpha)':<30}{results['LRA-AST']['fires'][idx]:<25}")
    print("-"*65)
print("=================================================================")
