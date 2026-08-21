import numpy as np
import time
import sys
sys.path.append('c:/Users/Welcome/TS_SCLF')

from rigorous_polar import (
    PolarCode, 
    decode_ts_sclf,
    compute_crc,
    POLY_CRC3,
    POLY_CRC5,
    check_crc
)

def encode_segmented_crc_param(info_payload, l_seg):
    """
    Parameterized segmented CRC encoder:
    - Segment 1: first l_seg payload bits + 3-bit CRC1.
    - Segment 2: remaining 48 - l_seg payload bits + 5-bit CRC2.
    Total info bits = 56.
    """
    u_I = np.zeros(56, dtype=int)
    # Segment 1: first l_seg payload bits
    u_I[0:l_seg] = info_payload[0:l_seg]
    u_I[l_seg : l_seg + 3] = compute_crc(u_I[0:l_seg], POLY_CRC3)
    # Segment 2: next 48 - l_seg payload bits
    u_I[l_seg + 3 : 51] = info_payload[l_seg:48]
    u_I[51:56] = compute_crc(u_I[0:51], POLY_CRC5)
    return u_I

print("=========================================================")
print("      GAP 5: JOINT CRC + THRESHOLD OPTIMIZATION SWEEP")
print("=========================================================")

snr_db = 1.0
num_trials = 1000
configs = [16, 24, 32]

ebno = 10**(snr_db / 10.0)
rate = 56.0 / 128.0
sigma = np.sqrt(1.0 / (2.0 * rate * ebno))

results = {}

for l_seg in configs:
    print(f"\nSimulating l_seg = {l_seg}...")
    
    # Initialize Codec with specific l_seg
    code = PolarCode(N=128, K=56, l_seg=l_seg)
    code.early_termination_count = 0
    
    fer_count = 0
    total_attempts = 0.0
    
    # Use a fixed seed for each configuration to ensure fair comparison
    np.random.seed(500)
    
    start_time = time.time()
    for trial in range(num_trials):
        payload = np.random.randint(0, 2, 48)
        u_I = encode_segmented_crc_param(payload, l_seg)
        x = code.encode(u_I)
        
        s = 1 - 2 * x
        noise = np.random.normal(0, sigma, code.N)
        r = s + noise
        y = 2 * r / (sigma**2)
        
        # Run LRA-AST decoder
        dec_u_I, success, attempts, _, _ = decode_ts_sclf(
            y, code, L=4, T=10, D1=16, D2=3, alpha=1.5, 
            apply_constraint=True, guard_band=8, use_lra_ast=True, gamma_lra=0.5
        )
        
        if not success or not np.array_equal(dec_u_I, u_I):
            fer_count += 1
        total_attempts += attempts
        
    elapsed = time.time() - start_time
    print(f"l_seg = {l_seg} completed in {elapsed:.2f} seconds.")
    
    results[l_seg] = {
        "fer": fer_count / num_trials,
        "attempts": total_attempts / num_trials,
        "fires": code.early_termination_count
    }

print("\n" + "="*70)
print(f"{'Segment 1 Payload Size (l_seg)':<35}{'FER':<10}{'Avg. Attempts':<15}{'ET Fires':<10}")
print("="*70)
for l_seg in configs:
    res = results[l_seg]
    print(f"{l_seg:<35}{res['fer']:<10.5f}{res['attempts']:<15.3f}{res['fires']:<10}")
print("="*70)
