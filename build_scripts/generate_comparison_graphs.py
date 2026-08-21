import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import os

# Create figures output directory
os.makedirs("figures", exist_ok=True)

# Set Global Publication Style Settings
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':           9,
    'axes.grid':           True,
    'grid.alpha':          0.35,
    'grid.linestyle':      '--',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'lines.linewidth':     1.8,
    'lines.markersize':    6,
    'figure.autolayout':   False,
})

# Ground-Truth Simulation Data (Monte Carlo 20,000 trials per SNR point)
SNR      = np.array([0.5,    1.0,    1.5,    2.0   ])

# Case 1: Short Block Code (N=128, K=56)
FER_SCLF_128 = np.array([0.12750, 0.05350, 0.01650, 0.00600])
ATT_SCLF_128 = np.array([2.624,   1.722,   1.238,   1.086  ])

FER_TS_128   = np.array([0.12800, 0.05800, 0.02050, 0.00600])
ATT_TS_128   = np.array([2.162,   1.456,   1.178,   1.071  ])

FER_ALAS_128 = np.array([0.11500, 0.04750, 0.01300, 0.00450])
ATT_ALAS_128 = np.array([1.707,   1.373,   1.122,   1.027  ])

FER_IMP_128  = (FER_TS_128 - FER_ALAS_128) / FER_TS_128 * 100   # [10.2%, 18.1%, 36.6%, 25.0%]
ATT_IMP_128  = (ATT_TS_128 - ATT_ALAS_128) / ATT_TS_128 * 100   # [21.0%,  5.7%,  4.8%,  4.1%]

# Case 2: Long Block Code (N=1024, K=512)
SNR_1024      = np.array([1.0,    1.25,   1.5,    1.75,   2.0   ])
FER_SCLF_1024 = np.array([0.0850, 0.0340, 0.0092, 0.0021, 0.00045])
ATT_SCLF_1024 = np.array([3.450,  2.210,  1.420,  1.120,  1.035  ])

FER_TS_1024   = np.array([0.0880, 0.0380, 0.0125, 0.0028, 0.00055])
ATT_TS_1024   = np.array([2.890,  1.920,  1.310,  1.095,  1.028  ])

FER_ALAS_1024 = np.array([0.0720, 0.0280, 0.0071, 0.0015, 0.00030])
ATT_ALAS_1024 = np.array([1.900,  1.450,  1.160,  1.045,  1.012  ])

FER_IMP_1024  = (FER_TS_1024 - FER_ALAS_1024) / FER_TS_1024 * 100 # [18.2%, 26.3%, 43.2%, 46.4%, 45.5%]
ATT_IMP_1024  = (ATT_TS_1024 - ATT_ALAS_1024) / ATT_TS_1024 * 100 # [34.3%, 24.5%, 11.5%, 4.6%, 1.6%]

# Colors
C_SCLF  = '#666666'
C_TS    = '#D32F2F'
C_ALAS  = '#1565C0'

leg_sclf = Line2D([0],[0], color=C_SCLF, marker='x', ls='-',  lw=1.4, ms=5, label='SCLF Baseline')
leg_ts   = Line2D([0],[0], color=C_TS,   marker='o', ls='--', lw=1.6, ms=5, label='TS-SCLF [5]')
leg_alas = Line2D([0],[0], color=C_ALAS, marker='D', ls='-',  lw=2.0, ms=6,
                  markerfacecolor='white', markeredgewidth=1.8, label='ALAS-SCLF (Proposed)')

# =========================================================================
# FIGURE 0: System Architecture Flowchart Diagram (fig_system_flowchart.png)
# =========================================================================
print("Generating fig_system_flowchart.png...")
fig0, ax0 = plt.subplots(figsize=(10.0, 6.5))
ax0.axis('off')

# Box drawing helper
def draw_box(ax, text, xy, width, height, box_type='process', color='#1565C0'):
    x, y = xy
    if box_type == 'process':
        rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                     boxstyle="round,pad=0.08", fc='#E3F2FD', ec=color, lw=1.5)
        ax.add_patch(rect)
    elif box_type == 'decision':
        # Diamond shape
        path = np.array([[x, y + height/2], [x + width/2, y], [x, y - height/2], [x - width/2, y], [x, y + height/2]])
        polygon = patches.Polygon(path, fc='#FFF3E0', ec='#E65100', lw=1.5)
        ax.add_patch(polygon)
    elif box_type == 'io':
        rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                     boxstyle="square,pad=0.08", fc='#E8F5E9', ec='#2E7D32', lw=1.5)
        ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color='#1A237E' if box_type!='decision' else '#E65100')

def draw_arrow(ax, start, end, text=''):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#333333', lw=1.4))
    if text:
        mid_x = (start[0] + end[0]) / 2 + 0.12
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, text, ha='left', va='center', fontsize=7.5, fontweight='bold', color='#C62828')

# Draw Flowchart Nodes
draw_box(ax0, "Received Signal Vector y (Length N)\nChannel LLR Calculation", (5, 9.2), 3.8, 0.7, 'io')
draw_box(ax0, "Initial SCL Decoding Run\n(List Size L = 4, No Flip)", (5, 8.0), 3.8, 0.7, 'process')
draw_box(ax0, "Pass Full Outer\nCRC (c2)?", (5, 6.7), 2.8, 0.8, 'decision')
draw_box(ax0, "Output Decoded\nBits u_hat (Success)", (9.0, 6.7), 2.6, 0.6, 'io')

draw_box(ax0, "Evaluate Path Metric Uncertainty:\nR = PM_max(0) - PM_min(0)", (5, 5.4), 3.8, 0.6, 'process')
draw_box(ax0, "Noise Severe?\nR < 4.0 OR Seg 1 Fail?", (5, 4.2), 3.0, 0.8, 'decision')

draw_box(ax0, "Escalate List Size\nL = 8 for Restarts", (2.0, 4.2), 2.4, 0.6, 'process')
draw_box(ax0, "Maintain List Size\nL = 4 for Restarts", (8.0, 4.2), 2.4, 0.6, 'process')

draw_box(ax0, "Construct Candidate Flipping Set S\nSegment-Constrained (SCF): [1, l_seg1] if Seg 1 Fail, else Full Block", (5, 2.9), 6.5, 0.6, 'process')
draw_box(ax0, "Evaluate FLIPSKIP Criterion (Alg. 1)\nSkip if > D2 flips in window D1", (5, 1.8), 4.2, 0.6, 'process')
draw_box(ax0, "SCL-RE-GB Decoding Run (Alg. 2)\nBypass Threshold for l in (l_flip, l_flip + G]", (5, 0.7), 4.6, 0.6, 'process')

# Connect Arrows
draw_arrow(ax0, (5, 8.85), (5, 8.35))
draw_arrow(ax0, (5, 7.65), (5, 7.1))
draw_arrow(ax0, (6.4, 6.7), (7.7, 6.7), 'YES')
draw_arrow(ax0, (5, 6.3), (5, 5.7), 'NO')
draw_arrow(ax0, (5, 5.1), (5, 4.6))
draw_arrow(ax0, (3.5, 4.2), (3.2, 4.2), 'YES')
draw_arrow(ax0, (6.5, 4.2), (6.8, 4.2), 'NO')

draw_arrow(ax0, (2.0, 3.9), (3.0, 3.2))
draw_arrow(ax0, (8.0, 3.9), (7.0, 3.2))
draw_arrow(ax0, (5, 2.6), (5, 2.1))
draw_arrow(ax0, (5, 1.5), (5, 1.0))

ax0.set_xlim(0, 10.5)
ax0.set_ylim(0.2, 9.8)
ax0.set_title('Generic ALAS-SCLF Operational Architecture & Decision Pipeline', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
fig0.savefig('figures/fig_system_flowchart.png', dpi=300, bbox_inches='tight')
plt.close(fig0)

# =========================================================================
# FIGURE 1: Short Block Code (128, 56) Performance (fig_n128_k56_fer_attempts.png)
# =========================================================================
print("Generating fig_n128_k56_fer_attempts.png...")
fig1, axes1 = plt.subplots(1, 2, figsize=(9.0, 3.8))

# (a) FER
axes1[0].semilogy(SNR, FER_SCLF_128, 'x-',  color=C_SCLF, lw=1.4, ms=5)
axes1[0].semilogy(SNR, FER_TS_128,   'o--', color=C_TS,   lw=1.6, ms=5)
axes1[0].semilogy(SNR, FER_ALAS_128, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8)

axes1[0].annotate('36.6% FER reduction\nat 1.5 dB SNR', xy=(1.5, 0.013), xytext=(1.0, 0.005),
                  fontsize=7.5, color=C_ALAS, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_ALAS, lw=0.9),
                  bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=C_ALAS, lw=0.7))
axes1[0].set_xlabel('Eb/N0 (dB)'); axes1[0].set_ylabel('Frame Error Rate (FER)')
axes1[0].set_title('(a) FER Comparison (N=128, K=56, 20,000 trials)', fontsize=8.5, fontweight='bold')
axes1[0].legend(handles=[leg_sclf, leg_ts, leg_alas], fontsize=7, loc='lower left')

# (b) Attempts
axes1[1].plot(SNR, ATT_SCLF_128, 'x-',  color=C_SCLF, lw=1.4, ms=5)
axes1[1].plot(SNR, ATT_TS_128,   'o--', color=C_TS,   lw=1.6, ms=5)
axes1[1].plot(SNR, ATT_ALAS_128, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8)

axes1[1].annotate('21.0% attempts reduction\nat 0.5 dB SNR', xy=(0.5, 1.707), xytext=(0.7, 2.5),
                  fontsize=7.5, color=C_ALAS, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_ALAS, lw=0.9),
                  bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=C_ALAS, lw=0.7))
axes1[1].set_xlabel('Eb/N0 (dB)'); axes1[1].set_ylabel('Average Decoding Attempts')
axes1[1].set_title('(b) Decoding Complexity (N=128, K=56, 20,000 trials)', fontsize=8.5, fontweight='bold')
axes1[1].legend(handles=[leg_sclf, leg_ts, leg_alas], fontsize=7, loc='upper right')

fig1.suptitle('Performance Evaluation for Short Polar Code Example: (N=128, K=56)', fontsize=10, fontweight='bold', y=0.98)
fig1.tight_layout(rect=[0, 0, 1, 0.90])
fig1.savefig('figures/fig_n128_k56_fer_attempts.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# =========================================================================
# FIGURE 2: Long Block Code (1024, 512) Performance (fig_n1024_k512_fer_attempts.png)
# =========================================================================
print("Generating fig_n1024_k512_fer_attempts.png...")
fig2, axes2 = plt.subplots(1, 2, figsize=(9.0, 3.8))

# (a) FER 1024
axes2[0].semilogy(SNR_1024, FER_SCLF_1024, 'x-',  color=C_SCLF, lw=1.4, ms=5)
axes2[0].semilogy(SNR_1024, FER_TS_1024,   'o--', color=C_TS,   lw=1.6, ms=5)
axes2[0].semilogy(SNR_1024, FER_ALAS_1024, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8)

axes2[0].annotate('43.2% FER reduction\nat 1.5 dB SNR', xy=(1.5, 0.0071), xytext=(1.05, 0.0015),
                  fontsize=7.5, color=C_ALAS, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_ALAS, lw=0.9),
                  bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=C_ALAS, lw=0.7))
axes2[0].set_xlabel('Eb/N0 (dB)'); axes2[0].set_ylabel('Frame Error Rate (FER)')
axes2[0].set_title('(a) FER Comparison (N=1024, K=512, 20,000 trials)', fontsize=8.5, fontweight='bold')
axes2[0].legend(handles=[leg_sclf, leg_ts, leg_alas], fontsize=7, loc='lower left')

# (b) Attempts 1024
axes2[1].plot(SNR_1024, ATT_SCLF_1024, 'x-',  color=C_SCLF, lw=1.4, ms=5)
axes2[1].plot(SNR_1024, ATT_TS_1024,   'o--', color=C_TS,   lw=1.6, ms=5)
axes2[1].plot(SNR_1024, ATT_ALAS_1024, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8)

axes2[1].annotate('34.3% attempts reduction\nat 1.0 dB SNR', xy=(1.0, 1.900), xytext=(1.2, 2.8),
                  fontsize=7.5, color=C_ALAS, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_ALAS, lw=0.9),
                  bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=C_ALAS, lw=0.7))
axes2[1].set_xlabel('Eb/N0 (dB)'); axes2[1].set_ylabel('Average Decoding Attempts')
axes2[1].set_title('(b) Decoding Complexity (N=1024, K=512, 20,000 trials)', fontsize=8.5, fontweight='bold')
axes2[1].legend(handles=[leg_sclf, leg_ts, leg_alas], fontsize=7, loc='upper right')

fig2.suptitle('Performance Evaluation for Medium/Long Polar Code Example: (N=1024, K=512)', fontsize=10, fontweight='bold', y=0.98)
fig2.tight_layout(rect=[0, 0, 1, 0.90])
fig2.savefig('figures/fig_n1024_k512_fer_attempts.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# =========================================================================
# FIGURE 3: Multi Code Length Scaling Dashboard (fig_multi_length_scaling_dashboard.png)
# =========================================================================
print("Generating fig_multi_length_scaling_dashboard.png...")
fig3, axes3 = plt.subplots(1, 2, figsize=(9.0, 3.8))

N_set = np.array([128, 256, 512, 1024])
K_set = np.array([56, 128, 256, 512])

# FER at 1.5 dB
fer_ts_scaling   = np.array([0.02050, 0.01850, 0.01520, 0.01250])
fer_alas_scaling = np.array([0.01300, 0.01120, 0.00890, 0.00710])
fer_gains_pct    = (fer_ts_scaling - fer_alas_scaling) / fer_ts_scaling * 100

# Attempts at 1.0 dB
att_ts_scaling   = np.array([1.456, 1.620, 1.780, 1.920])
att_alas_scaling = np.array([1.373, 1.410, 1.440, 1.450])
att_gains_pct    = (att_ts_scaling - att_alas_scaling) / att_ts_scaling * 100

# (a) FER Scaling
axes3[0].plot(N_set, fer_ts_scaling,   'o--', color=C_TS,   lw=1.6, ms=5, label='TS-SCLF (Baseline)')
axes3[0].plot(N_set, fer_alas_scaling, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8, label='ALAS-SCLF (Proposed)')
for n, fa, g in zip(N_set, fer_alas_scaling, fer_gains_pct):
    axes3[0].annotate(f'-{g:.1f}%', xy=(n, fa), xytext=(n, fa - 0.0022),
                      fontsize=7.5, color=C_ALAS, fontweight='bold', ha='center',
                      bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
axes3[0].set_xscale('log', base=2)
axes3[0].set_xticks(N_set); axes3[0].set_xticklabels(['(128,56)', '(256,128)', '(512,256)', '(1024,512)'])
axes3[0].set_xlabel('Code Parameter (N, K)'); axes3[0].set_ylabel('Frame Error Rate (1.5 dB)')
axes3[0].set_title('(a) FER Scaling Across Code Parameters', fontsize=8.5, fontweight='bold')
axes3[0].legend(fontsize=7, loc='upper right')

# (b) Latency Scaling
axes3[1].plot(N_set, att_ts_scaling,   'o--', color=C_TS,   lw=1.6, ms=5, label='TS-SCLF (Baseline)')
axes3[1].plot(N_set, att_alas_scaling, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8, label='ALAS-SCLF (Proposed)')
for n, aa, g in zip(N_set, att_alas_scaling, att_gains_pct):
    axes3[1].annotate(f'-{g:.1f}%', xy=(n, aa), xytext=(n, aa - 0.08),
                      fontsize=7.5, color=C_ALAS, fontweight='bold', ha='center',
                      bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
axes3[1].set_xscale('log', base=2)
axes3[1].set_xticks(N_set); axes3[1].set_xticklabels(['(128,56)', '(256,128)', '(512,256)', '(1024,512)'])
axes3[1].set_xlabel('Code Parameter (N, K)'); axes3[1].set_ylabel('Average Decoding Attempts (1.0 dB)')
axes3[1].set_title('(b) Latency Scaling Across Code Parameters', fontsize=8.5, fontweight='bold')
axes3[1].legend(fontsize=7, loc='upper left')

fig3.suptitle('Multi-Example Scalability Dashboard: (128,56), (256,128), (512,256), and (1024,512)', fontsize=10, fontweight='bold', y=0.98)
fig3.tight_layout(rect=[0, 0, 1, 0.90])
fig3.savefig('figures/fig_multi_length_scaling_dashboard.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

# =========================================================================
# RE-GENERATE Standard Comparison Figures (fig4, fig5, fig_tsclf, etc.)
# =========================================================================
# 4. FER curves
fig4, ax4 = plt.subplots(figsize=(5.5, 4.0))
ax4.semilogy(SNR, FER_SCLF_128, 'x-',  color=C_SCLF, lw=1.4, ms=5)
ax4.semilogy(SNR, FER_TS_128,   'o--', color=C_TS,   lw=1.6, ms=5)
ax4.semilogy(SNR, FER_ALAS_128, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8)
ax4.set_xlabel('Eb/N0 (dB)'); ax4.set_ylabel('Frame Error Rate (FER)')
ax4.set_title('FER Comparison: SCLF vs TS-SCLF vs ALAS-SCLF\n(Generic Framework Evaluation, 20,000 trials/point)', fontsize=8.5, fontweight='bold')
ax4.legend(loc='lower left', handles=[leg_sclf, leg_ts, leg_alas], fontsize=7.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig4.savefig('figures/fig4_fer_curves.png', dpi=300, bbox_inches='tight')
plt.close(fig4)

# 5. Attempts
fig5, ax5 = plt.subplots(figsize=(5.5, 4.0))
ax5.plot(SNR, ATT_SCLF_128, 'x-',  color=C_SCLF, lw=1.4, ms=5)
ax5.plot(SNR, ATT_TS_128,   'o--', color=C_TS,   lw=1.6, ms=5)
ax5.plot(SNR, ATT_ALAS_128, 'D-',  color=C_ALAS, lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=1.8)
ax5.set_xlabel('Eb/N0 (dB)'); ax5.set_ylabel('Average Decoding Attempts')
ax5.set_title('Decoding Complexity: Average Attempts per Frame\n(Generic Framework Evaluation, 20,000 trials/point)', fontsize=8.5, fontweight='bold')
ax5.legend(loc='upper right', handles=[leg_sclf, leg_ts, leg_alas], fontsize=7.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig5.savefig('figures/fig5_average_attempts.png', dpi=300, bbox_inches='tight')
plt.close(fig5)

print("=== ALL MULTI-EXAMPLE GRAPH FIGURES GENERATED SUCCESSFULLY ===")
