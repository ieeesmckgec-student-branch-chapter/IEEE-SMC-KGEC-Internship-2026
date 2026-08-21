import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from rigorous_polar import (
    PolarCode, 
    decode_ca_scl,
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
    print("=" * 65)
    print("  GENERATING ALAS-SCLF COMPARISON PLOTS  ")
    print("=" * 65)

    np.random.seed(42)
    N = 128
    K = 56
    snr_list = [0.5, 1.0, 1.5, 2.0]
    trials = 2000

    code_24 = PolarCode(N, K, l_seg=24) # TS version baseline
    code_16 = PolarCode(N, K, l_seg=16) # ALAS version proposed

    results = {s: {"fer": [], "attempts": []} for s in ["CA-SCL", "SCLF", "TS-SCLF", "ALAS-SCLF"]}

    for snr_db in snr_list:
        print(f"Simulating Eb/N0 = {snr_db:.1f} dB  ({trials} frames)...")
        ebno = 10**(snr_db / 10.0)
        rate = K / N
        sigma = np.sqrt(1.0 / (2.0 * rate * ebno))

        counts = {s: {"fer": 0, "attempts": 0.0} for s in results}

        for trial in range(trials):
            payload = np.random.randint(0, 2, K - 8)
            noise = np.random.normal(0, sigma, N)

            # ---- Standard 24-bit split segment (SCLF, TS-SCLF, CA-SCL) ----
            u_I_24 = encode_segmented_crc_generic(payload, K, l_seg=24)
            x_24 = code_24.encode(u_I_24)
            s_24 = 1 - 2 * x_24
            r_24 = s_24 + noise
            y_24 = 2 * r_24 / (sigma**2)

            # CA-SCL (L=4)
            dec_ca, succ_ca = decode_ca_scl(y_24, code_24, L=4)
            if not succ_ca or not np.array_equal(dec_ca, u_I_24):
                counts["CA-SCL"]["fer"] += 1
            counts["CA-SCL"]["attempts"] += 1.0

            # SCLF (L=4)
            dec_sclf, succ_sclf, att_sclf = decode_sclf(y_24, code_24, L=4, T=10, D1=16, D2=3)
            if not succ_sclf or not np.array_equal(dec_sclf, u_I_24):
                counts["SCLF"]["fer"] += 1
            counts["SCLF"]["attempts"] += att_sclf

            # TS-SCLF (L=4, alpha=1.0)
            dec_ts, succ_ts, att_ts, _, _ = decode_ts_sclf(
                y_24, code_24, L=4, T=10, D1=16, D2=3, alpha=1.0, apply_constraint=True, guard_band=8, use_lra_ast=False
            )
            if not succ_ts or not np.array_equal(dec_ts, u_I_24):
                counts["TS-SCLF"]["fer"] += 1
            counts["TS-SCLF"]["attempts"] += att_ts

            # ---- Asymmetric 16-bit split segment (ALAS-SCLF) ----
            u_I_16 = encode_segmented_crc_generic(payload, K, l_seg=16)
            x_16 = code_16.encode(u_I_16)
            s_16 = 1 - 2 * x_16
            r_16 = s_16 + noise
            y_16 = 2 * r_16 / (sigma**2)

            # ALAS-SCLF (L=4->8, l_seg=16, alpha=1.0)
            dec_alas, succ_alas, att_alas = decode_alas_sclf(y_16, code_16, T=10, D1=16, D2=3, alpha=1.0, guard_band=8)
            if not succ_alas or not np.array_equal(dec_alas, u_I_16):
                counts["ALAS-SCLF"]["fer"] += 1
            counts["ALAS-SCLF"]["attempts"] += att_alas

        for s in results:
            results[s]["fer"].append(counts[s]["fer"] / trials)
            results[s]["attempts"].append(counts[s]["attempts"] / trials)

    # -----------------------------------------------------------------------
    # Plotting (IEEE Publication Style with White Background)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=11)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)

    colors  = {"CA-SCL": "#555555", "SCLF": "#d62728", "TS-SCLF": "#1f77b4", "ALAS-SCLF": "#2ca02c"}
    markers = {"CA-SCL": "x",       "SCLF": "o",        "TS-SCLF": "s",       "ALAS-SCLF": "^"}
    lstyles = {"CA-SCL": "--",      "SCLF": "-",        "TS-SCLF": "-",       "ALAS-SCLF": "-"}

    # ---- Left Plot: FER ----
    for scheme in ["CA-SCL", "SCLF", "TS-SCLF", "ALAS-SCLF"]:
        ax1.semilogy(snr_list, results[scheme]["fer"],
                     label=scheme, color=colors[scheme],
                     marker=markers[scheme], linestyle=lstyles[scheme],
                     linewidth=2.0, markersize=8)

    ax1.set_xlabel("$E_b/N_0$ (dB)", fontsize=13)
    ax1.set_ylabel("Frame Error Rate (FER)", fontsize=13)
    ax1.set_title("Frame Error Rate (FER) Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5, color="gray")
    ax1.legend(fontsize=11, facecolor="white", labelcolor="black", edgecolor="#cccccc")

    # ---- Right Plot: Decoding Complexity ----
    for scheme in ["SCLF", "TS-SCLF", "ALAS-SCLF"]:
        ax2.plot(snr_list, results[scheme]["attempts"],
                 label=scheme, color=colors[scheme],
                 marker=markers[scheme], linestyle=lstyles[scheme],
                 linewidth=2.0, markersize=8)

    ax2.set_xlabel("$E_b/N_0$ (dB)", fontsize=13)
    ax2.set_ylabel("Average Decoding Attempts", fontsize=13)
    ax2.set_title("Decoding Complexity (Average Attempts)", fontsize=14, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5, color="gray")
    ax2.legend(fontsize=11, facecolor="white", labelcolor="black", edgecolor="#cccccc")

    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polar_comparison_results.png")
    plt.savefig(out_img, dpi=300, facecolor=fig.get_facecolor())
    print(f"\nSaved comparison plot -> {out_img}")
    print("=" * 65)
