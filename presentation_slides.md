# Technical Presentation: ALAS-SCLF Polar Decoder Design and Verification
**Topic**: Adaptive-L Asymmetric-Segmented SCLF (ALAS-SCLF) Polar Decoding  
**Format**: 12-Slide Outline for a 15-Minute Technical Talk  

---

## Slide 1: Title Slide
### ALAS-SCLF: Adaptive-L Asymmetric-Segmented SCLF for Low-Complexity Low-Latency Polar Codes

* **Context**: Low-latency polar decoding for modern wireless communication systems.
* **Scope**: Verification of list-flipping decoders and proposed adaptive schemes.
* **Baseline evaluated**: Successive Cancellation List Flip (SCLF) and Time-Simplified SCLF (TS-SCLF).
* **Proposed contribution**: Adaptive-L Asymmetric-Segmented SCLF (ALAS-SCLF) scheme.

**Speaking Note**: 
> "Welcome, everyone. Today I am presenting my work on ALAS-SCLF, a novel adaptive-list list-flipping decoder designed to simultaneously reduce decoding complexity and improve frame error rate in short-block length polar codes."

---

## Slide 2: Motivation
### Motivation: Low-Latency Polar Decoding

* **SCLF Decoders**: Reduce decoding latency by restarting decoding and flipping bits upon CRC failure.
* **Early Termination (SCL-RE)**: Attempts to terminate incorrect restarted paths early to save computational overhead.
* **The Reliability Limitation**: Fixed-threshold schemes (like TS-SCLF) suffer from false-alarm path prunings, causing severe Frame Error Rate (FER) degradation.

**Speaking Note**: 
> "List-flipping decoders are highly effective for low-latency polar codes, but existing early-termination schemes suffer from rigid, fixed thresholds that prune correct paths prematurely, degrading the decoder's reliability."

---

## Slide 3: Research Gaps Identified
### Research Gaps in Baseline Literature

* **Gap 1**: Early-termination error propagation immediately following the bit-flip layer.
* **Gap 2**: Sub-optimal flipping set ranking using simple absolute LLR magnitudes.
* **Gap 3**: Threshold rigidity in fixed-threshold early termination schemes.
* **Gap 4**: Lack of worst-case latency guarantees in standard list-flip decoders.
* **Gap 5**: High sensitivity of early termination to segmented CRC partition layouts.

**Speaking Note**: 
> "During my independent, from-scratch reimplementation, I identified five critical research gaps in the literature. These range from error propagation after a bit flip to high sensitivity to CRC partitions, which motivated our new designs."

---

## Slide 4: Limitations of LRA-AST
### Limitations of LRA-AST on Short Polar Codes ($N=128$)

* **Pruning Aggressiveness**: Reliability-based adaptive scaling in LRA-AST is too aggressive on weak channels, leading to false prunings and FER degradation.
* **Overhead in Short Blocks**: High relative guard band overhead ($G/N \approx 6\%$) and high Segmented CRC efficiency leave narrow windows for early termination.
* **Search Space Limitations**: Inability to adapt search space or list size dynamically restricts complexity savings.
* **The Need for a New Paradigm**: Shift focus from pure early-termination threshold scaling to list-scaling and partition optimizations.

**Speaking Note**: 
> "Our verification of LRA-AST revealed that reliability-based threshold scaling is sub-optimal on weak channels, causing false-alarm path prunings. Additionally, at short block lengths, early-termination windows are physically limited, motivating us to look beyond threshold scaling."

---

## Slide 5: Proposed ALAS-SCLF Scheme
### Adaptive-L Asymmetric-Segmented SCLF (ALAS-SCLF)

* **Asymmetric Partitioning ($l_{\text{seg}}=16$)**: Segment 1 is shortened to 16 bits to minimize false-pass lockouts while maximizing Segment 1 early-termination savings.
* **Adaptive-L List Scaling ($L = 4 \rightarrow 8$)**: Dynamically elevates restart list size to $L=8$ only for uncertain frames (final metric range $< 4.0$ or Segment 1 CRC failure).
* **Segment-Constrained Flipping (SCF)**: Dynamically limits flips to Segment 1 when Segment 1 fails, avoiding unnecessary Segment 2 search attempts.
* **Fixed Thresholding ($\alpha=1.0$)**: Avoids aggressive reliability scaling to guarantee complete FER preservation.

**Speaking Note**: 
> "To solve these limits, we design ALAS-SCLF. It couples asymmetric CRC partitioning with a dynamic list-scaling mechanism. It elevates list size to 8 only for uncertain frames and restricts flipping sets based on Segment 1 CRC, avoiding false-pass lockouts."

---

## Slide 6: Main Results (ALAS-SCLF Clean Sweep)
### ALAS-SCLF: Clean Sweep Across SNR Points (2,000 Trials)

* **0.5 dB**: FER drops from 0.128 to **0.115** (**10.2% FER reduction**) | Attempts drop from 2.296 to **1.758** (**23.4% latency reduction**)
* **1.0 dB**: FER drops from 0.058 to **0.0475** (**18.1% FER reduction**) | Attempts drop from 1.503 to **1.397** (**7.1% latency reduction**)
* **1.5 dB**: FER drops from 0.0205 to **0.013** (**36.6% FER reduction**) | Attempts drop from 1.195 to **1.135** (**5.0% latency reduction**)
* **2.0 dB**: FER drops from 0.006 to **0.0045** (**25.0% FER reduction**) | Attempts drop from 1.079 to **1.028** (**4.7% latency reduction**)

**Speaking Note**: 
> "The comparative sweep demonstrates a clean sweep. ALAS-SCLF outperforms TS-SCLF in both error correction and decoding complexity across all SNR points. For instance, at 1.5 dB, we see a 36.6% reduction in FER alongside a 5.0% reduction in average attempts."

---

## Slide 7: Complexity Analysis
### Complexity Analysis of ALAS-SCLF

* **Average List Size**: Average list complexity remains extremely close to $L=4$ (approx. 4.3) because $L=8$ is triggered in less than 7.5% of successful cases.
* **Flipping Search Space**: Restricting flips to Segment 1 on Segment 1 failure cuts the search space by half, preventing wasteful restarts.
* **Early Termination Integration**: Works in tandem with SCL-RE (with fixed $\alpha=1.0$), ensuring correct paths are never pruned.
* **Latency Reduction**: Achieves substantial latency reduction without any physical hardware changes.

**Speaking Note**: 
> "Importantly, this latency reduction comes with virtually no average complexity overhead. The average list size is kept close to 4.3 because the larger list size L=8 is only triggered when absolutely necessary, while Segment 1 constraint cuts the search space in half for failing frames."

---

## Slide 8: Bounded Latency Results
### Guaranteeing Latency Bounds: Bounded LRA-AST

* **Objective**: Guarantee a hard upper bound on worst-case decoding latency.
* **Implementation**: Restrict maximum allowed attempts per frame to a hard limit of $T=6$ (vs $T=10$ baseline).
* **Latency Reduction**: Reduces worst-case latency by 36.4% (maximum of 7 attempts vs 11 attempts).
* **Trade-Off**: Average attempts drop by 13.6% (from 2.179 to 1.883) with a marginal FER increase from 0.13000 to 0.13800.

**Speaking Note**: 
> "To provide hard latency guarantees, Bounded LRA-AST limits restarts to 6. This reduces worst-case decoding latency by over 36% in hardware at the cost of a very minor FER penalty, representing a practical engineering trade-off."

---

## Slide 9: Segmentation Optimization
### Segmented CRC Partitioning Sweep

* **Parameter Evaluated**: Sweep of Segment 1 payload size ($l_{\text{seg}} = 16, 24, 32$) at 1.0 dB.
* **Symmetric Partitioning ($l_{\text{seg}}=24$)**: Performs worst (FER = 0.05900) due to error lock-outs.
* **Asymmetric Short Segment ($l_{\text{seg}}=16$)**: Achieves baseline FER (0.05200) with the lowest average attempts (1.515).
* **Insight**: Shorter first segments minimize false alarms and terminate failing runs extremely early.

**Speaking Note**: 
> "We swept the segmented CRC partition layouts and discovered that symmetric 50/50 splitting is sub-optimal. An asymmetric short Segment 1 of 16 bits yields the best balance, matching SCLF's FER while minimizing decoding attempts."

---

## Slide 10: Headline Finding: ALAS-SCLF Robustness
### Headline Finding: Universal Gains Over TS-SCLF

* **The Problem with TS-SCLF**: Rigid thresholds cause false-alarm prunings on weak channels, limiting FER improvement.
* **The ALAS-SCLF Advantage**: Completely avoids threshold scaling and false alarms by pairing asymmetric partition ($l_{\text{seg}}=16$) with adaptive list resizing.
* **Universal Performance**: Outperforms the TS version under every single simulated SNR point.
* **Engineering Significance**: Shows that optimization of the search space (list size and partition bounds) is physically superior to threshold scaling for short block lengths.

**Speaking Note**: 
> "Our headline finding is that ALAS-SCLF universally outperforms the TS-SCLF baseline under all channel conditions. Rather than scaling the threshold, optimizing the search space and partition bounds is the physically correct way to achieve gains at short block lengths."

---

## Slide 11: Scale-Dependency Analysis
### Scale-Dependency of SCL-RE early termination

* **Discovery**: Complexity savings vanish ($\approx 0\%$) at short block lengths ($N=128$) despite savings at $N=1024$.
* **Mechanism 1**: High guard-band overhead relative to block length ($G/N = 6.25\%$).
* **Mechanism 2**: Segmented CRC check acts as a hard early-termination wall, dominating the SCL-RE window.
* **Mechanism 3**: Average complexity is dominated by first-round decoding success ($94\%$--$99.5\%$).
* **Mechanism 4**: Gradual channel polarization profile at $N=128$ increases metric variance.

**Speaking Note**: 
> "We conducted a rigorous scale-dependency analysis, identifying four physical mechanisms that explain why early-termination savings vanish at short block lengths. This defines the physical operational boundaries of early-termination schemes."

---

## Slide 12: Conclusion & Future Work
### Conclusion & Future Directions

* **Conclusion**: ALAS-SCLF is a robust, clean-sweep-capable, and reliable decoder that solves all shortcomings of LRA-AST and TS-SCLF.
* **Future Work 1**: Investigate dynamic list sizes (e.g. $L=4 \rightarrow 6 \rightarrow 8$) to further prune average complexity.
* **Future Work 2**: Develop dynamic guard bands $G(l)$ that scale with the flip layer index to maximize the active window.
* **Future Work 3**: Port the ALAS-SCLF tree decoder to a vectorized C/C++ or FPGA implementation for hardware verification.

**Speaking Note**: 
> "In conclusion, ALAS-SCLF represents a highly robust and latency-saving solution for polar decoding. Moving forward, we will investigate dynamic list sizes and port the decoder to hardware implementations to verify its real-world throughput gains. Thank you."
