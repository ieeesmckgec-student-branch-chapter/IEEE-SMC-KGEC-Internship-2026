# Reproduction Guide

Everything needed to reproduce MedMNIST-Defer on a fresh machine. Two paths:
**sample mode** needs no download and finishes in seconds; **full mode**
downloads 574 MB and reproduces every number in the paper.

---

## 0. Requirements

| | Minimum | Used for the published results |
|---|---|---|
| Python | 3.9 | 3.13 |
| RAM | 8 GB | 32 GB |
| Disk | 100 MB (sample) / 1.5 GB (full) | — |
| GPU | not required | NVIDIA RTX PRO 4500 Blackwell |

CPU-only works for both modes. Full mode on CPU takes roughly 1-2 hours instead
of 5 minutes.

---

## 1. Install

```bash
git clone https://github.com/stabgan/medmnist-defer.git
cd medmnist-defer

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### GPU support

The default `torch` wheel may be CPU-only depending on your platform. For CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Match the URL to your CUDA version. Check what you got:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. Sample mode (no download)

The repository ships class-stratified subsets of all 12 datasets — 600 training
and 400 test images each, 13 MB total — so the full pipeline runs immediately.

```bash
python scripts/verify_setup.py
python scripts/run_benchmark.py --mode sample
python scripts/make_figures.py --results results/benchmark_sample.json \
                               --outdir results/figures_sample
```

Or all at once:

```bash
./run_all.sh
```

Expected output:

```
MedMNIST-Defer | mode=sample backbone=resnet18 device=cuda
Datasets: 12 | epochs=3 seed=42

[ 1/12] bloodmnist      acc=0.263 acc@50%=0.430 gain=+0.166 p=0.0000 (1.4s)
...
Mean gain @50% coverage (entropy): +0.147
Datasets with significant gain: 11/12
```

The final step compares your output against `results/golden/benchmark_sample.json`
and should report all comparisons within tolerance.

> **Sample-mode numbers are not results.** With 600 training images and 3 epochs,
> absolute accuracy is far below the full-data values. This mode verifies that
> training, all five deferral policies, the bootstrap and the figures all execute.
> No number from sample mode appears in the paper.

---

## 3. Full mode (reproduces the paper)

### 3.1 Download

```bash
python scripts/download_data.py
```

Fetches 12 archives from the official Zenodo record into `data/raw/` and verifies
each against the MD5 published in the `medmnist` package. The script refuses to
proceed on a mismatch, and re-running it skips files that are already verified.

| Dataset | Size |
|---|---:|
| pathmnist | 196 MB |
| tissuemnist | 119 MB |
| chestmnist | 79 MB |
| octmnist | 52 MB |
| organamnist | 37 MB |
| bloodmnist | 34 MB |
| dermamnist | 19 MB |
| organsmnist | 16 MB |
| organcmnist | 15 MB |
| pneumoniamnist | 4 MB |
| retinamnist | 3 MB |
| breastmnist | 0.5 MB |
| **Total** | **574 MB** |

Subsets only:

```bash
python scripts/download_data.py --datasets pathmnist bloodmnist dermamnist
```

### 3.2 Verify

```bash
python scripts/verify_setup.py --mode full
```

Expect 12 `[OK]` lines and a total of about 574 MB.

### 3.3 Run

```bash
python scripts/run_benchmark.py --mode full
```

Roughly 5.5 minutes on one modern GPU. Writes `results/benchmark_full.json`.

Expected console output:

```
[ 1/12] bloodmnist      acc=0.797 acc@50%=0.961 gain=+0.164 p=0.0000 (9.3s)
[ 2/12] breastmnist     acc=0.635 acc@50%=0.782 gain=+0.132 p=0.0000 (0.8s)
[ 3/12] chestmnist      acc=0.530 acc@50%=0.710 gain=+0.180 p=0.0000 (49.7s)
[ 4/12] dermamnist      acc=0.707 acc@50%=0.897 gain=+0.192 p=0.0000 (5.5s)
[ 5/12] octmnist        acc=0.644 acc@50%=0.740 gain=+0.096 p=0.0000 (47.8s)
[ 6/12] organamnist     acc=0.875 acc@50%=0.997 gain=+0.122 p=0.0000 (26.1s)
[ 7/12] organcmnist     acc=0.869 acc@50%=0.991 gain=+0.123 p=0.0000 (10.9s)
[ 8/12] organsmnist     acc=0.730 acc@50%=0.939 gain=+0.210 p=0.0000 (11.6s)
[ 9/12] pathmnist       acc=0.651 acc@50%=0.812 gain=+0.161 p=0.0000 (53.0s)
[10/12] pneumoniamnist  acc=0.821 acc@50%=0.958 gain=+0.137 p=0.0000 (3.0s)
[11/12] retinamnist     acc=0.460 acc@50%=0.660 gain=+0.201 p=0.0000 (1.2s)
[12/12] tissuemnist     acc=0.564 acc@50%=0.751 gain=+0.187 p=0.0000 (107.4s)

Mean gain @50% coverage (entropy): +0.160
Datasets with significant gain: 12/12
```

### 3.4 Figures and comparison

```bash
python scripts/make_figures.py --results results/benchmark_full.json
python scripts/compare_to_golden.py --candidate results/benchmark_full.json
```

Or the whole sequence:

```bash
./run_all.sh full
```

---

## 4. How close should your numbers be?

**On the same machine, runs are exactly reproducible.** The random state is reset
at the start of every dataset, so results also do not depend on how many datasets
you run or in what order:

```bash
python scripts/run_benchmark.py --mode full --datasets breastmnist retinamnist
python scripts/run_benchmark.py --mode full --datasets retinamnist breastmnist
```

Both produce identical numbers, and both match the corresponding rows of a full
12-dataset run.

**Across machines, exact reproduction is not expected**, even with a fixed seed.
cuDNN picks different kernels on different GPUs and floating-point reductions
happen in different orders, so trained weights diverge slightly.

`compare_to_golden.py` applies a 5-point absolute tolerance and prints every
difference. Interpretation:

| Difference | Meaning |
|---|---|
| Within 1-2 points | Normal hardware and library variation |
| 2-8 points on breastmnist / retinamnist | Expected; their test sets hold 156 and 400 images |
| More than 5 points across many datasets | Check seed, epochs, and that you used `--mode full` |
| Sign of the gain flips | Something is wrong; open an issue |

These qualitative conclusions should hold on any hardware:

- significant gains at 50% coverage on all or nearly all 12 datasets
- all five policies clustered within roughly one point of each other
- temperature scaling roughly neutral for deferral, clearly positive for ECE
- breastmnist significant at 50% coverage but not at 90%

Absolute per-dataset accuracies are the least stable quantity in the benchmark.
See [paper/07_discussion.md](paper/07_discussion.md) §7.4 for a measured estimate of
how much they move under a change of random state alone.

---

## 5. Variations

### Different backbone

```bash
python scripts/run_benchmark.py --mode full --backbone simplecnn
```

### Longer training

```bash
python scripts/run_benchmark.py --mode full --epochs 30
```

Expect higher absolute accuracy and smaller deferral gains — a better-fitted model
has less headroom.

### Selected datasets

```bash
python scripts/run_benchmark.py --mode full --datasets pathmnist dermamnist bloodmnist
```

Note that the Wilcoxon signed-rank test needs at least 5 datasets and reports
`null` below that.

### Custom output path

```bash
python scripts/run_benchmark.py --mode full --output results/my_run.json
python scripts/compare_to_golden.py --candidate results/my_run.json
```

---

## 6. Higher resolutions (MedMNIST+)

The published results use 28x28, the original MedMNIST v2 format. The same
subsets are also published at 64, 128 and 224 pixels. Extending the benchmark
across resolutions is the most valuable open follow-up (see
[paper/07_discussion.md](paper/07_discussion.md) §7.3).

Approximate download sizes for all 12 subsets:

| Resolution | Total | Notes |
|---|---:|---|
| 28x28 | 574 MB | what this benchmark uses |
| 64x64 | ~2-3 GB | |
| 128x128 | ~15 GB | |
| 224x224 | ~40 GB | |

To fetch them:

```bash
pip install medmnist
python -m medmnist download --size=64
```

Files land in `~/.medmnist/`. Adapting the benchmark requires two changes:
copy or link the `*_64.npz` files into `data/raw/` under the plain names, and
confirm the model's first convolution and pooling still suit the larger input
(ResNet-18 handles all four sizes without modification). Runtime scales roughly
with pixel count.

---

## 7. Regenerating the committed sample data

Only needed if you want to change the sample sizes.

```bash
python scripts/download_data.py            # full archives must be present
python scripts/build_sample_data.py
```

Sizes are controlled by `SAMPLE_TRAIN_PER_DATASET` and
`SAMPLE_TEST_PER_DATASET` in `scripts/config.py`. Sampling is class-stratified
with seed 42 so rare classes survive; the multi-label subset uses a uniform draw.

---

## 8. Troubleshooting

**`FileNotFoundError: Missing .../metadata.json`**

```bash
python scripts/download_data.py --metadata-only
```

**`FileNotFoundError: Missing data/raw/pathmnist.npz`**

You are in full mode without the download. Run `python scripts/download_data.py`,
or use `--mode sample`.

**`RuntimeError: MD5 mismatch`**

A truncated or corrupted download. Delete the file in `data/raw/` and retry.

**`ValueError: Expected more than 1 value per channel when training`**

A batch-norm layer received a single-sample batch. The benchmark already guards
against this with `drop_last=True` plus a size check; if you see it, you have
likely lowered the batch size below 2 or modified the loader.

**CUDA out of memory**

```bash
python scripts/run_benchmark.py --mode full --epochs 8   # then lower batch size
```

Edit `PROFILES["full"].batch_size` in `scripts/config.py` (try 64 or 32). Results
will shift slightly because batch-norm statistics change.

**Already have MedMNIST downloaded elsewhere**

```bash
ln -s ~/.medmnist/pathmnist.npz data/raw/pathmnist.npz
```

Then `python scripts/download_data.py` verifies checksums without re-downloading.

**Bootstrap is slow**

Lower `bootstrap_n` in `PROFILES` in `scripts/config.py`. Bootstrap resampling,
not training, dominates runtime on the large subsets: TissueMNIST spends most of
its 107 seconds re-sorting 47,280 confidence values 500 times per coverage level
per policy.

---

## 9. Where every published number comes from

| Paper location | Source |
|---|---|
| §6.1 aggregate table | `results/golden/benchmark_full.json` → `aggregate` |
| §6.2 Table 1 | → `per_dataset.*.policies.entropy.acc_at_coverage` |
| §6.2 confidence intervals | → `per_dataset.*.policies.entropy.deferral_gain.delta@50%` |
| §6.3 Table 2 | → `per_dataset.*.n_test` and `confidence_intervals` |
| §6.4 Table 3 | → `aggregate.per_policy` |
| §6.5 Table 4 | → `per_dataset.*.ece_uncalibrated`, `ece_temperature_scaled`, `temperature` |
| §6.6 correlation | → `aggregate.correlation_ece_vs_gain_at_50pct` |
| §5.4 runtimes | → `per_dataset.*.runtime_sec`, `runtime_sec` |
| Figures 1-4 | `python scripts/make_figures.py` |

Inspect any value directly:

```bash
python -c "
import json
d = json.load(open('results/golden/benchmark_full.json'))
print(json.dumps(d['aggregate'], indent=2))
"
```
