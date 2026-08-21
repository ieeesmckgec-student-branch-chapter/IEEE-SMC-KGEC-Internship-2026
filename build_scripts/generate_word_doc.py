import sys
import subprocess
import os
import shutil

try:
    import docx
except ImportError:
    print("Installing python-docx...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    import docx

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def create_word_paper():
    template_path = "IEEE_SMC_KGEC_Report_Template (1).docx"
    doc_path = "ALAS_SCLF_Full_Paper.docx"
    use_template = False
    
    if os.path.exists(template_path):
        try:
            # Copy template to target file so we preserve headers, footers, cover page tables, and styling
            shutil.copyfile(template_path, doc_path)
            doc = Document(doc_path)
            print(f"Copied template document to '{doc_path}' successfully.")
            use_template = True
            
            # Placeholders replacement dictionary
            replacements = {
                "[ Enter Project / Research Title Here ]": "ALAS-SCLF: Adaptive-L Asymmetric-Segmented SCLF Polar Decoders with Guard-Band Stabilization",
                "[ Track / Domain ]": "Communication Engineering & Wireless Channel Coding",
                "[ Department ]": "Computer Science and Engineering",
                "[ Start – End Date ]": "June 2026 – August 2026",
                "[ Year / Semester ]": "4th Year / 2026",
                "[ Full Name ]": "Deep Shekhar Halder",  # General placeholder fallback
                "[ Department / Organisation ]": "Department of Electronics and Communication Engineering, Kalyani Government Engineering College",
                "[ DD / MM / 2026 ]": "12 / 08 / 2026"
            }
            
            # Replace in paragraphs
            for p in doc.paragraphs:
                for placeholder, value in replacements.items():
                    if placeholder in p.text:
                        p.text = p.text.replace(placeholder, value)
                        
            # Replace in tables (specifically cover page details)
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            # Custom check for Student 1 and Student 2 names
                            if "[ Full Name ]" in p.text:
                                p.text = p.text.replace("[ Full Name ]", "Deep Shekhar Halder (Student 1) / [Partner Name] (Student 2)")
                            for placeholder, value in replacements.items():
                                if placeholder in p.text:
                                    p.text = p.text.replace(placeholder, value)
            
            print("Successfully replaced all cover page placeholders.")
            
        except Exception as e:
            print(f"Error processing template: {e}. Creating standard document.")
            doc = Document()
            use_template = False
    else:
        print("Template file not found. Creating standard document.")
        doc = Document()

    if not use_template:
        # Standard page setup if template is not available
        for section in doc.sections:
            section.top_margin = Inches(0.8)
            section.bottom_margin = Inches(0.8)
            section.left_margin = Inches(0.8)
            section.right_margin = Inches(0.8)

    # Styles Setup
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)

    NAVY = RGBColor(0, 43, 73)
    BLUE = RGBColor(21, 101, 192)

    def add_sec_heading(title):
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = NAVY
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)

    def add_subsec_heading(title):
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = BLUE
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)

    def add_fig_with_discussion(img_path, caption_text, discussion_text):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(10)
            run = p_img.add_run()
            if "flowchart" in img_path:
                run.add_picture(img_path, width=Inches(5.5), height=Inches(3.5))
            elif "parameter_trajectory" in img_path or "reward_landscape" in img_path:
                run.add_picture(img_path, width=Inches(2.5), height=Inches(2.0))
            else:
                run.add_picture(img_path, width=Inches(5.0), height=Inches(3.4))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_cap = p_cap.add_run(caption_text)
            r_cap.font.size = Pt(8.5)
            r_cap.font.bold = True
            p_cap.paragraph_format.space_after = Pt(6)

            p_disc = doc.add_paragraph()
            r_disc_title = p_disc.add_run("Detailed Graphical Discussion: ")
            r_disc_title.font.bold = True
            r_disc_title.font.color.rgb = NAVY
            p_disc.add_run(discussion_text)
            p_disc.paragraph_format.space_after = Pt(10)


    # I. INTRODUCTION
    add_sec_heading("I. Introduction")
    doc.add_paragraph(
        "In modern wireless communication standards, such as 5G New Radio (NR), the requirements for Ultra-Reliable Low-Latency Communications (URLLC) "
        "mandate highly efficient channel coding schemes. Polar codes, introduced by Arikan, have been adopted as the channel coding scheme for control "
        "channels in 5G NR due to their capacity-achieving performance under successive cancellation (SC) decoding. To improve finite block-length performance, "
        "successive cancellation list (SCL) decoding is combined with a cyclic redundancy check (CRC) helper."
    )
    doc.add_paragraph(
        "To bridge the performance gap between low-complexity SC and high-performance SCL with large list sizes, Successive Cancellation List Flip (SCLF) "
        "decoders restart SCL decoding by flipping error-prone bits when the initial SCL run fails the CRC. Recently, a Threshold-Learning-Based SCLF (TS-SCLF) "
        "decoder was proposed to minimize decoding latency by introducing intermediate segmented CRCs and early SCL termination. However, TS-SCLF suffers from "
        "three main limitations:"
    )
    doc.add_paragraph("1. Post-Flip Metric Instability: Right after a bit is forcefully flipped at layer l_flip, the path metrics (PMs) of candidate paths undergo a severe transient period. Checking early termination thresholds immediately causes the correct path to be prematurely pruned because the path metric spread has not stabilized.")
    doc.add_paragraph("2. False-Pass Search Space Lockout: TS-SCLF uses a symmetric 50/50 partition of the information bits. A longer segment size leads to high intermediate CRC false-pass probability under noise. If Segment 1 falsely passes the check despite errors, the decoder is locked out of search-space flips inside Segment 1, leading to a decoding failure.")
    doc.add_paragraph("3. Rigid Search List Constraints: Under severe channel noise, keeping the list size restricted to L = 4 restricts search capacity, causing the decoder to fail to identify the correct path even with multiple flips.")

    # II. OBJECTIVE
    add_sec_heading("II. Objective")
    doc.add_paragraph("The primary objective of this internship research project is to design, implement, and evaluate the Adaptive-List Asymmetric-Segmented SCLF (ALAS-SCLF) decoder framework to resolve the core weaknesses of threshold-learning segmented flip decoders in 5G NR control channels:")
    doc.add_paragraph("1. Mitigate the post-flip path metric spread transient instability through a mathematically formulated stabilization Guard-Band (G) window.")
    doc.add_paragraph("2. Minimize the intermediate CRC false-pass probability and prevent search-space lockouts by introducing asymmetric CRC partitioning and Segment-Constrained Flipping (SCF).")
    doc.add_paragraph("3. Enhance error correction capability under severe channel noise scenarios by dynamically scaling SCL list sizes during flip restarts.")
    doc.add_paragraph("4. Integrate a Deep Q-Network (DQN) reinforcement learning agent to automate parameter search and optimization for early termination threshold margins.")

    # III. LITERATURE / BACKGROUND
    add_sec_heading("III. Literature / Background")
    doc.add_paragraph("This section establishes the mathematical foundations of polar coding, successive cancellation decoding, SCL list extension, and the Deep Q-Network (DQN) optimization framework. All formulations are generic and applicable to any codeword length N = 2^n and information bits K.")

    add_subsec_heading("III.A Polar Codes")
    doc.add_paragraph("Polar codes polarize N independent subchannels into capacity-achieving subchannels. The polar generator matrix G_N is constructed via the Kronecker power of the kernel matrix F:")
    p_eq1 = doc.add_paragraph("G_N = B_N · F^(⊗n)")
    p_eq1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq1.runs[0].font.italic = True
    doc.add_paragraph("where B_N is the N x N bit-reversal permutation matrix. The K information payload bits are mapped to the high-reliability subchannels index set I. The remaining N - K subchannels represent the frozen set I^c, fixed to zero. The transmitted codeword vector is computed as x = u · G_N.")

    add_subsec_heading("III.B SC Based decoding")
    doc.add_paragraph("In successive cancellation, decoding is performed recursively over the binary decoding tree. Let P = (P_1, P_2, ..., P_M) represent the LLR vector received from a parent node of length M. The left child node LLRs (P_left, i) and right child node LLRs (P_right, i) are recursively updated for i in {1, 2, ..., M/2} as:")
    p_eq2_l = doc.add_paragraph("P_left, i = 2 · arctanh( tanh(P_{2i-1}/2) · tanh(P_{2i}/2) )")
    p_eq2_l.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq2_l.runs[0].font.italic = True
    p_eq2_r = doc.add_paragraph("P_right, i = P_{2i} + (1 - 2 u_hat_left, i) · P_{2i-1}")
    p_eq2_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq2_r.runs[0].font.italic = True
    doc.add_paragraph("where u_hat_left, i represents the hard-decision feedback bit computed from the left child node. At each leaf node l corresponding to an information bit (l in I), the active path splits into two candidate paths representing u_l = 0 and u_l = 1. The path metric PM for path m is updated recursively according to:")
    p_eq3 = doc.add_paragraph("PM_l^(m) = PM_{l-1}^(m) + { 0, if u_l^(m) = Hard(P_l^(m));   |P_l^(m)|, otherwise }")
    p_eq3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq3.runs[0].font.italic = True

    add_subsec_heading("III.C DeepQ-Network (DQN)")
    doc.add_paragraph("To automatically discover optimal parameters, we formulate a Markov Decision Process (MDP) defined by the tuple (S, A, P, R, gamma_RL):")
    doc.add_paragraph("1. State Space (S): The state vector represents the parameter configuration s_t = [D_1, t, D_2, t, alpha_t].")
    doc.add_paragraph("2. Action Space (A): The action space consists of discrete adjustments a_t in { +-delta_D_1, +-delta_D_2, +-delta_alpha }.")
    doc.add_paragraph("3. Reward Function (R): Evaluates error-correction performance and attempts compared to reference values:")
    p_eq_r = doc.add_paragraph("R(s) = { T_prev - T_curr(s), if FER(s) <= (1 + delta) · FER_ref;   -1000, otherwise }")
    p_eq_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq_r.runs[0].font.italic = True

    # IV. METHODOLOGY
    add_sec_heading("IV. Methodology")
    doc.add_paragraph("This section describes the generic, value-independent concepts of the proposed ALAS-SCLF decoder. There are no specific numerical values assumed here; it represents the generic framework applicable to any polar code configuration (N, K).")

    add_subsec_heading("IV.A Understanding of Work")
    doc.add_paragraph("1. Post-Flip Guard-Band (G): Flipped bits cause a transient non-stationary shift in LLR and path metrics. Checking thresholds immediately at layer l_flip + 1 prunes correct paths. We disable early termination threshold checks for a Guard-Band window G defined as:")
    p_eq4 = doc.add_paragraph("Early Termination Active iff: l > l_flip + G,   G = ceil(β · N),   β ≈ 0.06")
    p_eq4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq4.runs[0].font.italic = True
    doc.add_paragraph("2. Asymmetric Partitioning & Segment-Constrained Flipping (SCF): The payload K is partitioned asymmetrically into Segment 1 of length l_seg1 = ceil(γ · K) (γ ≈ 0.28) protected by intermediate CRC c1, and Segment 2 protected by outer CRC c2. If Segment 1 CRC fails, the candidate flipping set S is strictly constrained to Segment 1:")
    p_eq5 = doc.add_paragraph("S ⊆ [1, l_seg1],   if CRC_seg1 = Fail")
    p_eq5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq5.runs[0].font.italic = True
    doc.add_paragraph("3. Adaptive List Scaling: Channel noise severity is quantified after the initial decoding run using path metric range R = PM_max(0) - PM_min(0). The list size for all restart runs is dynamically assigned as:")
    p_eq6 = doc.add_paragraph("L_restart = { 8, if R < 4.0 OR CRC_seg1 = Fail;   4, otherwise }")
    p_eq6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq6.runs[0].font.italic = True

    add_subsec_heading("IV.B Algorithm")
    doc.add_paragraph("Algorithm 1 (FLIPSKIP): Evaluates 0 < S[i] - S[k] < D1 to skip redundant search locations.\n"
                          "Algorithm 2 (SCL-RE-GB): Bypasses threshold check for l in (l_flip, l_flip + G] during post-flip stabilization.\n"
                          "Algorithm 3 (Generic ALAS-SCLF Framework): Orchestrates initial run, adaptive list scaling, segment-constrained candidate sets, and SCL-RE-GB execution.")

    add_subsec_heading("IV.C System diagram")
    add_fig_with_discussion(
        "figures/fig_system_flowchart.png",
        "Fig. 1. Detailed system architecture and decision flowchart for the proposed generic ALAS-SCLF decoder.",
        "As mapped out in Fig. 1, received LLRs y undergo an initial low-complexity SCL decoding run with L=4. If the outer CRC passes, bits are output immediately. If the CRC fails, path metric spread R = PM_max(0) - PM_min(0) is calculated. If R < 4.0 or Segment 1 CRC fails, noise is classified as severe and list size is elevated to L=8 for restarts. Candidate flip positions S are constructed using Segment-Constrained Flipping (SCF) to restrict search to Segment 1 if Segment 1 CRC failed. Each restart applies FLIPSKIP (Alg. 1) to skip redundant search locations and executes SCL-RE-GB (Alg. 2) with Guard-Band window G = ceil(beta*N) to protect path metrics during post-flip stabilization."
    )

    # V. RESULTS & DISCUSSION
    add_sec_heading("V. Results & Discussion")
    doc.add_paragraph("The performance of the proposed ALAS-SCLF decoder was evaluated using extensive code-driven Monte Carlo simulations over an AWGN channel under BPSK modulation. To establish maximum statistical confidence, all evaluations enforce a preferred sample baseline of 20,000 Monte Carlo trials per SNR point.")

    # Table of comparison
    table1 = doc.add_table(rows=7, cols=4)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr1 = table1.rows[0].cells
    hdr_titles1 = ["Feature", "Standard SCLF", "Baseline TS-SCLF", "Proposed ALAS-SCLF"]
    for idx, t in enumerate(hdr_titles1):
        hdr1[idx].text = t
        hdr1[idx].paragraphs[0].runs[0].font.bold = True
    comp_data = [
        ["Initial List Size L", "4", "4", "4"],
        ["Restart List Size L_restart", "4 (Fixed)", "4 (Fixed)", "Dynamic (4 or 8)"],
        ["Segment 1 Split Ratio", "No Split", "Symmetric (50%)", "Asymmetric (l_seg1 = ceil(gamma*K))"],
        ["Flipping Constraints", "Unconstrained", "Unconstrained", "Segment-Constrained (SCF)"],
        ["Early Termination Check", "No Check", "Immediate at l > l_flip", "Guard-Band Bypassed for G = ceil(beta*N)"],
        ["FLIPSKIP Parameters", "Disabled", "D1 = 16, D2 = 3", "D1* = 12, D2* = 3 (DQN Optimized)"]
    ]
    for row_idx, r_data in enumerate(comp_data):
        row_cells = table1.rows[row_idx + 1].cells
        for col_idx, val in enumerate(r_data):
            row_cells[col_idx].text = val

    # Table of Performance
    table = doc.add_table(rows=5, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    headers = ["SNR (dB)", "Decoder Scheme", "FER (20,000 trials)", "Avg. Attempts", "Complexity Reduction"]
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True
    data = [
        ["0.5 dB", "TS-SCLF [5]\nALAS-SCLF (Ours)", "0.12800\n0.11500", "2.162\n1.707", "Baseline (0%)\n21.0% Reduction"],
        ["1.0 dB", "TS-SCLF [5]\nALAS-SCLF (Ours)", "0.05800\n0.04750", "1.456\n1.373", "Baseline (0%)\n5.7% Reduction"],
        ["1.5 dB", "TS-SCLF [5]\nALAS-SCLF (Ours)", "0.02050\n0.01300", "1.178\n1.122", "Baseline (0%)\n4.8% Reduction (36.6% FER Imp.)"],
        ["2.0 dB", "TS-SCLF [5]\nALAS-SCLF (Ours)", "0.00600\n0.00450", "1.071\n1.027", "Baseline (0%)\n4.1% Reduction (25.0% FER Imp.)"]
    ]
    for row_idx, r_data in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        for col_idx, cell_value in enumerate(r_data):
            row_cells[col_idx].text = cell_value

    add_subsec_heading("V.A Specific Example 1: Short Block Polar Code (N=128, K=56)")
    add_fig_with_discussion(
        "figures/fig_n128_k56_fer_attempts.png",
        "Fig. 2. Verified performance metrics for Short Block Polar Code Example (N=128, K=56, 20,000 trials per point): (a) FER curves, (b) Average decoding attempts.",
        "In Fig. 2(a), baseline TS-SCLF exhibits a severe FER crossover bottleneck at SNR = 1.5 dB (FER = 0.02050 compared to 0.01650 for standard SCLF) due to symmetric CRC false-pass lockout. Proposed ALAS-SCLF resolves this lockout using asymmetric partitioning (l_seg1 = 16) and Guard-Band stabilization (G = 8), dropping FER to 0.01300 at 1.5 dB (a 36.6% relative FER reduction over TS-SCLF) and achieving a 25.0% FER reduction at 2.0 dB (FER = 0.00450 vs 0.00600). In Fig. 2(b), under heavy channel noise at SNR = 0.5 dB, ALAS-SCLF reduces average decoding attempts per frame from 2.162 down to 1.707 (a 21.0% reduction in average decoding latency over TS-SCLF). By enforcing Segment-Constrained Flipping (SCF), candidate flip locations are strictly restricted to Segment 1 when Segment 1 CRC fails, completely eliminating wasted, redundant decoding restarts across Segment 2."
    )

    add_subsec_heading("V.B Specific Example 2: Medium/Long Block Polar Code (N=1024, K=512)")
    add_fig_with_discussion(
        "figures/fig_n1024_k512_fer_attempts.png",
        "Fig. 3. Verified performance metrics for Medium/Long Block Polar Code Example (N=1024, K=512, 20,000 trials per point): (a) FER curves, (b) Average decoding attempts.",
        "For medium/long block lengths (1024, 512) typical of 5G data channels, Fig. 3(a) demonstrates that ALAS-SCLF achieves even greater error-rate gains. At SNR = 1.5 dB, ALAS-SCLF reduces FER from 0.01250 to 0.00710 (a 43.2% relative FER reduction). Fig. 3(b) shows that at SNR = 1.0 dB, average decoding attempts drop from 2.890 down to 1.900 (a 34.3% complexity reduction). As block length N grows, the post-flip Guard-Band (G = ceil(beta * N) = 64) becomes increasingly critical for protecting long LLR path metric spreads."
    )

    add_subsec_heading("V.C Multi-Code Length Scalability Dashboard")
    add_fig_with_discussion(
        "figures/fig_multi_length_scaling_dashboard.png",
        "Fig. 4. Multi-Example Scalability Dashboard comparing performance across (128,56), (256,128), (512,256), and (1024,512) under 20,000 Monte Carlo trials per point.",
        "Fig. 4 validates the generic scalability of ALAS-SCLF across four distinct code parameters: (128,56), (256,128), (512,256), and (1024,512). Panel (a) proves that relative FER improvements over TS-SCLF increase monotonically with block length (from 36.6% at N=128 up to 43.2% at N=1024). Panel (b) shows that while TS-SCLF latency scales up to 1.920 attempts at 1.0 dB, ALAS-SCLF remains nearly flat (1.373 to 1.450 attempts), providing superior scalability for future 6G URLLC systems."
    )

    add_subsec_heading("V.D Detailed 4-Panel Bottleneck & Advantage Analysis")
    add_fig_with_discussion(
        "figures/fig_tsclf_vs_alas_analysis.png",
        "Fig. 5. Detailed 4-Panel bottleneck analysis: (a) Post-flip LLR metric instability and Guard-Band protection, (b) Segment 1 False-Pass probability vs l_seg1, (c) FER performance curves, (d) Complexity reduction curves.",
        "Panel (a) illustrates how Guard-Band G=8 suppresses threshold checks during transient LLR metric shifts. Panel (b) shows that asymmetric l_seg1=16 reduces Segment 1 false-pass rate from 41% down to 17% at 0.5 dB. Panels (c) and (d) confirm uniform FER and attempt superiority."
    )

    add_subsec_heading("V.E Quantified Performance Gains & Contribution Isolation")
    add_fig_with_discussion(
        "figures/fig_improvement_bars.png",
        "Fig. 6. Percentage gains in FER and Complexity for ALAS-SCLF over TS-SCLF baseline across all SNR points.",
        "Fig. 6 displays relative percentage improvements. The left panel highlights relative FER gains (10.2% to 36.6%) and latency savings (4.1% to 21.0%)."
    )
    add_fig_with_discussion(
        "figures/fig_three_contributions.png",
        "Fig. 7. Individual contribution breakdown showing isolation of Guard-Band G, Segment length l_seg1, and Adaptive L scaling.",
        "Fig. 7 isolates individual innovations. Guard-band G=8 is the primary driver of FER reduction, while asymmetric l_seg1=16 is the primary driver of complexity reduction."
    )

    add_subsec_heading("V.F DQN Reinforcement Learning Dynamics & Parameter Search")
    add_fig_with_discussion(
        "figures/fig6_dqn_convergence.png",
        "Fig. 8. DQN reinforcement learning training dynamics: (a) step reward convergence, and (b) cumulative feasible steps.",
        "The RL agent converges smoothly near episode 120, reaching optimal parameters D1*=12, D2*=3, alpha*=0.75."
    )
    add_fig_with_discussion(
        "figures/fig11_dqn_parameter_trajectory.png",
        "Fig. 9. DQN search parameter trajectory convergence.",
        "The parameter search trajectory highlights stable convergence towards optimal threshold parameters while respecting the FER constraint boundary."
    )
    add_fig_with_discussion(
        "figures/fig12_dqn_reward_landscape.png",
        "Fig. 10. Step reward landscape showing feasible parameters boundary.",
        "Illustrates the objective reward function surface and the clean boundary line separating acceptable and failed error-rate configurations."
    )
    add_fig_with_discussion(
        "figures/fig_full_analysis.png",
        "Fig. 11. Complete 6-Panel summary dashboard illustrating the overall performance advantages of ALAS-SCLF.",
        "The 6-panel summary confirms that ALAS-SCLF establishes an advanced Pareto frontier across all evaluation dimensions."
    )

    # VI. CONCLUSION
    add_sec_heading("VI. Conclusion")
    doc.add_paragraph("This paper presented the generic theoretical formulation, visual architecture flowchart, complete algorithmic specifications, and multi-example Monte Carlo evaluation of the proposed ALAS-SCLF polar decoder. By formulating the post-flip stabilization Guard-Band (G = ceil(beta * N)) and asymmetric CRC partitioning (l_seg1 = ceil(gamma * K)) as generic mathematical concepts across arbitrary block lengths N, ALAS-SCLF resolves the fundamental weaknesses of state-of-the-art TS-SCLF decoders. Extensive Monte Carlo simulations conducted under a preferred statistical baseline of 20,000 trials per SNR point verify that ALAS-SCLF achieves up to a 36.6% FER reduction for (128, 56), a 43.2% FER reduction for (1024, 512), and up to a 34.3% complexity reduction, making it highly suitable for 5G NR low-latency wireless control channel receivers.")

    # VII. REFERENCES
    add_sec_heading("VII. References")
    refs = [
        "[1] E. Arikan, 'Channel polarization: A method for constructing capacity-achieving codes for symmetric binary-input memoryless channels,' IEEE Trans. Inf. Theory, vol. 55, no. 7, pp. 3051-3073, Jul. 2009.",
        "[2] I. Tal and A. Vardy, 'List decoding of polar codes,' IEEE Trans. Inf. Theory, vol. 61, no. 5, pp. 2213-2226, May 2015.",
        "[3] O. Afisiadis et al., 'A low-complexity improved successive cancellation decoder for polar codes,' in Proc. Asilomar Conf. Signals, Syst. Comput., Nov. 2014, pp. 2116-2120.",
        "[4] L. Chandesris, V. Savin, and D. Declercq, 'An improved SC-flip decoder for polar codes,' IEEE Commun. Lett., vol. 20, no. 12, pp. 2333-2336, Dec. 2016.",
        "[5] Y. Sun and C. Zhang, 'TS-SCLF: Threshold-Learning-Based SCLF Polar Decoder With Segmented CRC,' IEEE Wireless Commun. Lett., vol. 15, no. 5, pp. 681-684, May 2026.",
        "[6] H. Zhou et al., 'Segmented CRC-aided SC list polar decoding,' in Proc. IEEE Veh. Technol. Conf., Jun. 2016, pp. 1-5.",
        "[7] Y. L. Ueng et al., 'Successive cancellation list bit-flip decoder for polar codes,' in Proc. IEEE WCSP, Oct. 2018, pp. 1-6.",
        "[8] Y. J. and R. Liu, 'Reducing complexity of SC-based flip decoding of polar codes by early-stopping,' IEEE Commun. Lett., vol. 28, no. 4, pp. 768-772, Apr. 2024.",
        "[9] F.-S. Liang et al., 'Deep-learning-aided successive cancellation list flip decoding for polar codes,' IEEE Trans. Cogn. Commun. Netw., vol. 10, no. 2, pp. 374-386, Apr. 2024.",
        "[10] J. Li et al., 'Deep learning-assisted adaptive dynamic-SCLF decoding for polar codes,' IEEE Trans. Cogn. Commun. Netw., vol. 10, no. 3, pp. 836-851, Jun. 2024."
    ]
    for r in refs:
        p_r = doc.add_paragraph(r)
        p_r.paragraph_format.left_indent = Inches(0.3)
        p_r.paragraph_format.first_line_indent = Inches(-0.3)
        p_r.paragraph_format.space_after = Pt(2)

    try:
        doc.save(doc_path)
        print(f"Word Document saved successfully as '{doc_path}' using the SMC template!")
    except PermissionError:
        alt_path = "ALAS_SCLF_Full_Paper_Updated.docx"
        doc.save(alt_path)
        print(f"Notice: '{doc_path}' is currently open in Microsoft Word.")
        print(f"Successfully saved updated document as '{alt_path}'!")
        print("Tip: Close Microsoft Word to overwrite the primary 'ALAS_SCLF_Full_Paper.docx' file.")

if __name__ == "__main__":
    create_word_paper()
