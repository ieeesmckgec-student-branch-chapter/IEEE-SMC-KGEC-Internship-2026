# PhantomSense

PhantomSense is a computer-vision component for assisting underwater search and
rescue teams. The project uses side-scan sonar imagery and an object-detection
model to highlight possible submerged human bodies, helping investigators
prioritize areas for closer inspection.

The accompanying internship report describes a broader system that combines
sonar, GPS/IoT devices, chemical sensing, a server/API, alerts, and optional
drone verification. This repository contains the current local image-inference
prototype: `src/predict.py` and the included YOLO weights.

> PhantomSense is decision-support software. It does not replace trained human
> investigators, confirm a death, or perform recovery operations. Water
> conditions, sonar artifacts, occlusion, and image quality can produce false
> positives or false negatives.

## Features

- Runs Ultralytics YOLO inference on a single image.
- Uses CUDA automatically when an NVIDIA GPU is available; otherwise uses CPU.
- Searches for the newest trained checkpoint under `src/runs/train/`.
- Falls back to the bundled `src/yolo11n.pt` baseline when no trained run is found.
- Prints each detection's class, confidence, and bounding-box coordinates.
- Saves an annotated image to `src/outputs/`.

## Repository Structure

```text
.
├── docs/
│   └── IEEE_SMC_SBC_KGEC_2.pdf   # Internship report
├── src/
│   ├── predict.py                 # Inference entry point
│   ├── yolo11n.pt                 # Fallback model used by predict.py
│   └── yolo26n.pt                 # Additional bundled model
├── requirements.txt
└── README.md
```

The repository does not include a dataset, a sample image, or a `runs/`
training directory. Add those locally when needed; generated outputs are
written to `src/outputs/`.

## Requirements

- Python 3.10 or newer
- `pip`
- Optional: an NVIDIA GPU with a compatible PyTorch/CUDA installation

Install the project dependencies from the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running Inference

`predict.py` currently uses the image name in `IMAGE_NAME`. Place that image at
`src/images/4df43a1e-3377112912_10AUG25_1458_00.png`, or change `IMAGE_NAME` to
the name of an image in `src/images/` or to an existing path.

From the repository root, run:

```bash
python src/predict.py
```

The script reports the selected device and model, then prints detections such
as class name, confidence, and `[x1, y1, x2, y2]` coordinates. The annotated
result is saved as:

```text
src/outputs/marked_<input-stem>.png
```

For example, the default image produces
`src/outputs/marked_4df43a1e-3377112912_10AUG25_1458_00.png`.

## Model Selection

The script checks `src/runs/train/` for experiment directories and selects the
most recently modified directory containing `weights/best.pt`. This makes a
locally trained model take precedence automatically. If no trained checkpoint
is found, it uses `src/yolo11n.pt`.

Inference settings in `src/predict.py` are currently:

| Setting | Value |
| --- | ---: |
| Confidence threshold | `0.20` |
| IoU threshold | `0.45` |
| Image size | `1280` |
| Output annotation | Enabled by OpenCV save |

## Project Direction

The report presents the following intended development path:

1. Use sonar imagery to identify suspicious underwater regions.
2. Add chemical sensing for blood and decomposing organic material as
	supporting evidence.
3. Connect sensors through a server, API, dashboard, and alert system.
4. Add drone-based verification and more localized IoT deployment.

Only the first computer-vision stage is represented by the executable code in
this checkout. The chemical sensing, backend, dashboard, alerting, and drone
features remain future work.

## Reference

See [`docs/IEEE_SMC_SBC_KGEC_2.pdf`](docs/IEEE_SMC_SBC_KGEC_2.pdf) for the full
internship report, system proposal, limitations, and methodology. The report
references the AquaScan 1K Side Scan Sonar Dataset:

<https://zenodo.org/records/18771165>
