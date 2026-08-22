"""Central configuration for the MedMNIST-Defer benchmark.

Every path and hyper-parameter used by the experiments lives here so that a
reproduction run can be inspected and modified from a single file.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
RAW_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "metadata.json"

RESULTS_DIR = REPO_ROOT / "results"
GOLDEN_DIR = RESULTS_DIR / "golden"
FIGURES_DIR = RESULTS_DIR / "figures"

# The 12 two-dimensional MedMNIST v2 datasets, in alphabetical order.
DATASETS = (
    "bloodmnist",
    "breastmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist",
)

# Human-readable modality names used in tables and figures.
MODALITY_NAMES = {
    "bloodmnist": "Blood cell microscopy",
    "breastmnist": "Breast ultrasound",
    "chestmnist": "Chest X-ray (multi-label)",
    "dermamnist": "Dermatoscopy",
    "octmnist": "Retinal OCT",
    "organamnist": "Abdominal CT (axial)",
    "organcmnist": "Abdominal CT (coronal)",
    "organsmnist": "Abdominal CT (sagittal)",
    "pathmnist": "Colorectal histopathology",
    "pneumoniamnist": "Chest X-ray (pediatric)",
    "retinamnist": "Fundus photography",
    "tissuemnist": "Kidney cortex microscopy",
}

SEED = 42

# Coverage levels at which selective-prediction accuracy is reported.
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)

# Deferral policies compared in the benchmark.
DEFERRAL_METHODS = ("entropy", "max_prob", "margin", "temp_entropy", "mc_dropout")

# Fraction of the training split held out to fit the temperature parameter.
VAL_FRACTION = 0.1

# Number of stochastic forward passes for the MC-dropout policy.
MC_DROPOUT_PASSES = 10

# Bootstrap resamples for the 95% confidence intervals on accuracy at coverage.
BOOTSTRAP_N = 500

# Coverage levels that receive bootstrap confidence intervals.
BOOTSTRAP_COVERAGES = (0.9, 0.7, 0.5)

# Bins used for the expected-calibration-error estimate.
ECE_BINS = 15


class RunProfile:
    """Hyper-parameters for one of the two supported run modes."""

    def __init__(self, name: str, epochs: int, batch_size: int, lr: float, bootstrap_n: int):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.bootstrap_n = bootstrap_n


# `sample` runs on the tiny subsets committed to the repository so that anyone
# can execute the whole pipeline in a couple of minutes on a CPU.
# `full` reproduces the numbers reported in the paper.
PROFILES = {
    "sample": RunProfile("sample", epochs=3, batch_size=64, lr=1e-3, bootstrap_n=200),
    "full": RunProfile("full", epochs=8, batch_size=128, lr=1e-3, bootstrap_n=BOOTSTRAP_N),
}

BACKBONES = ("resnet18", "simplecnn")
DEFAULT_BACKBONE = "resnet18"

# Sizes of the committed sample subsets, per dataset and split.
SAMPLE_TRAIN_PER_DATASET = 600
SAMPLE_TEST_PER_DATASET = 400
