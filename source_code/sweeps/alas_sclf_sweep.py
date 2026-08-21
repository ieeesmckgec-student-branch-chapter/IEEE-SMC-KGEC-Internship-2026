import sys
import numpy as np
import time

sys.path.append('c:/Users/Welcome/TS_SCLF')

from rigorous_polar import (
    PolarCode, 
    decode_sclf, 
    decode_ts_sclf,
    decode_alas_sclf,
    POLY_CRC5,
    check_crc
)

def encode_segmented_crc_generic(info_payload, K, l_seg):
    POLY_CRC3 = np.array([1, 0, 1, 1], dtype=int)
    u_I = np.zeros(K, dtype=int)
    # Segment 1: first l_seg bits
    u_I[0:l_seg] = info_payload[0:l_seg]
    u_I[l_seg : l_seg + 3] = compute_crc(u_I[0:l_seg], POLY_CRC3)
    # Segment 2: next K - 8 - l_seg bits
    u_I[l_seg + 3 : K - 5] = info_payload[l_seg : K - 8]
    u_I[K - 5 : K] = compute_crc(u_I[0 : K - 5], POLY_CRC5)
    return u_I

def compute_crc(msg, poly):
    deg = len(poly) - 1
    padded = np.concatenate([msg, np.zeros(deg, dtype=int)])
    for i in range(len(msg)):
        if padded[i] == 1:
            padded[i : i + len(poly)] ^= poly
    return padded[-deg:]

if __name__ == "__main__":
    print("=================================================================")
    print("           ALAS-SCLF COMPARATIVE SIMULATION SWEEP                ")
    print("=================================================================")
    
    np.random.seed(42)
    N = 128
    K = 56
    snr_points = [0.5, 1.0, 1.5, 2.0]
    num_trials = 2000
    
    code_24 = PolarCode(N, K, l_seg=24) # TS version baseline
    code_16 = PolarCode(N, K, l_seg=16) # ALAS version proposed
    
    results = {}
    
    with open("alas_sclf_sweep_log.txt", "w") as log_file:
        log_file.write("SNR_dB,Scheme,FER,Avg_Attempts\n")
        
        for snr_db in snr_points:
            print(f"\nEvaluating SNR = {snr_db:.1f} dB (2,000 trials)...")
            ebno = 10**(snr_db / 10.0)
            rate = K / N
            sigma = np.sqrt(1.0 / (2.0 * rate * ebno))
            
            counts = {
                "SCLF (L=4, l_seg=24)": {"fer": 0, "attempts": 0.0},
                "TS-SCLF (L=4, l_seg=24, alpha=1.0)": {"fer": 0, "attempts": 0.0},
                "ALAS-SCLF (L=4->8, l_seg=16, alpha=1.0)": {"fer": 0, "attempts": 0.0}
            }
            
            for trial in range(num_trials):
                payload = np.random.randint(0, 2, K - 8)
                
                # 1. Standard 3/5 CRC split (l_seg=24)
                u_I_24 = encode_segmented_crc_generic(payload, K, l_seg=24)
                x_24 = code_24.encode(u_I_24)
                s_24 = 1 - 2 * x_24
                noise = np.random.normal(0, sigma, N)
                r_24 = s_24 + noise
                y_24 = 2 * r_24 / (sigma**2)
                
                # SCLF
                dec1, succ1, att1 = decode_sclf(y_24, code_24, L=4, T=10, D1=16, D2=3)
                if not succ1 or not np.array_equal(dec1, u_I_24):
                    counts["SCLF (L=4, l_seg=24)"]["fer"] += 1
                counts["SCLF (L=4, l_seg=24)"]["attempts"] += att1
                
                # TS-SCLF
                dec2, succ2, att2, _, _ = decode_ts_sclf(
                    y_24, code_24, L=4, T=10, D1=16, D2=3, alpha=1.0, apply_constraint=True, guard_band=8, use_lra_ast=False
                )
                if not succ2 or not np.array_equal(dec2, u_I_24):
                    counts["TS-SCLF (L=4, l_seg=24, alpha=1.0)"]["fer"] += 1
                counts["TS-SCLF (L=4, l_seg=24, alpha=1.0)"]["attempts"] += att2
                
                # 2. Asymmetric 3/5 CRC split (l_seg=16)
                u_I_16 = encode_segmented_crc_generic(payload, K, l_seg=16)
                x_16 = code_16.encode(u_I_16)
                s_16 = 1 - 2 * x_16
                r_16 = s_16 + noise
                y_16 = 2 * r_16 / (sigma**2)
                
                # ALAS-SCLF
                dec3, succ3, att3 = decode_alas_sclf(y_16, code_16, T=10, D1=16, D2=3, alpha=1.0, guard_band=8)
                if not succ3 or not np.array_equal(dec3, u_I_16):
                    counts["ALAS-SCLF (L=4->8, l_seg=16, alpha=1.0)"]["fer"] += 1
                counts["ALAS-SCLF (L=4->8, l_seg=16, alpha=1.0)"]["attempts"] += att3
                
            results[snr_db] = {}
            for name, data in counts.items():
                fer = data["fer"] / num_trials
                att = data["attempts"] / num_trials
                results[snr_db][name] = {"fer": fer, "attempts": att}
                log_file.write(f"{snr_db},{name},{fer:.5f},{att:.3f}\n")
                print(f"  {name:<42} FER = {fer:.5f} | Avg. Attempts = {att:.3f}")
                
    print("\n" + "="*85)
    print("                     ALAS-SCLF SIMULATION SWEEP SUMMARY")
    print("="*85)
    print(f"{'SNR (dB)':<10}{'Scheme':<42}{'FER':<15}{'Avg. Attempts':<15}")
    print("-"*85)
    for snr_db in snr_points:
        for name in results[snr_db]:
            fer = results[snr_db][name]["fer"]
            att = results[snr_db][name]["attempts"]
            print(f"{snr_db:<10.1f}{name:<42}{fer:<15.5f}{att:<15.3f}")
        print("-"*85)
    print("="*85)
