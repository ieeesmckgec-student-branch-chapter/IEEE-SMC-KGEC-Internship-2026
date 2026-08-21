# Pneumonia Detection from Chest X-ray Images using Deep Learning

A deep-learning based computer vision project for detecting pneumonia from chest X-ray images using transfer learning with ResNet50, DenseNet121, and EfficientNetB0. The system also provides explainable predictions using Grad-CAM and Grad-CAM++ visualization.

## Project Overview

This project develops an automated and explainable pneumonia detection system using chest X-ray images.

The workflow includes dataset acquisition, image preprocessing, data augmentation, exploratory data analysis, transfer learning, model training, evaluation, model comparison, Grad-CAM/Grad-CAM++ explainability, and interactive image prediction.

The project performs binary classification between:

* `NORMAL`
* `PNEUMONIA`

The input X-ray images are resized to `224 × 224 × 3` and normalized before being passed to the deep learning models.

## Project Information

| **Field**      | **Details**                     |
| -------------- | ------------------------------- |
| Group          | Group 3                         |
| Project Type   | Individual Project              |
| Project Author | **Isha Agarwal**                |
| Domain         | Computer Vision & Healthcare AI |
| Task           | Binary Image Classification     |
| Framework      | TensorFlow 2.x / Keras          |
| Explainability | Grad-CAM & Grad-CAM++           |



## Objectives

* Detect pneumonia from chest X-ray images.
* Classify X-ray images into NORMAL and PNEUMONIA categories.
* Apply image preprocessing and data augmentation.
* Implement transfer learning using multiple CNN architectures.
* Compare ResNet50, DenseNet121, and EfficientNetB0.
* Evaluate models using Accuracy, Precision, Recall, F1-score, and AUC.
* Generate confusion matrices and ROC curves.
* Identify the best-performing model based on AUC.
* Provide explainable predictions using Grad-CAM and Grad-CAM++.
* Allow users to upload a custom chest X-ray image for prediction.
* Display the predicted class and confidence score.
* Export the final best-performing model in multiple formats.

## Dataset

**Dataset**: Chest X-ray Pneumonia Dataset

**Dataset Source**: Kaggle — `paultimothymooney/chest-xray-pneumonia`

The notebook automatically downloads the dataset using `kagglehub`, so the complete dataset does not need to be uploaded to the GitHub repository.

### Dataset Directory Structure

The expected dataset structure is:

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The recorded notebook run contains:

| **Dataset Split** | **Images** |
| ----------------- | ---------: |
| Training          |      5,216 |
| Validation        |         16 |
| Testing           |        624 |
| Classes           |          2 |

The class mapping used by the project is:

```text
NORMAL    → 0
PNEUMONIA → 1
```

## Methodology

```text
Chest X-ray Dataset
        ↓
Dataset Download using KaggleHub
        ↓
Dataset Directory Resolution
        ↓
Image Preprocessing
        ↓
Image Resizing (224 × 224)
        ↓
Pixel Normalization
        ↓
Data Augmentation
        ↓
Exploratory Data Analysis
        ↓
Transfer Learning
        ↓
┌───────────────┬────────────────┬──────────────────┐
│   ResNet50    │  DenseNet121   │  EfficientNetB0  │
└───────────────┴────────────────┴──────────────────┘
        ↓
Model Training
        ↓
Early Stopping + Learning Rate Reduction
        ↓
Model Checkpointing
        ↓
Test Set Evaluation
        ↓
Accuracy + Precision + Recall + F1 + AUC
        ↓
Confusion Matrix + ROC Curve
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Grad-CAM & Grad-CAM++
        ↓
Interactive X-ray Prediction
        ↓
Final Model Export
```

## Image Preprocessing

All X-ray images are resized to:

```text
224 × 224 × 3
```

Pixel values are normalized using:

```python
rescale = 1./255
```

The training dataset additionally uses data augmentation techniques including:

* Rotation
* Zoom
* Width shifting
* Height shifting
* Horizontal flipping
* Brightness adjustment
* Nearest-neighbour filling

Validation and test images are only rescaled without augmentation.

### Main Configuration

| **Parameter**       | **Value**     |
| ------------------- | ------------- |
| Image Size          | 224 × 224 × 3 |
| Batch Size          | 32            |
| Classification Type | Binary        |
| Training Images     | 5,216         |
| Validation Images   | 16            |
| Test Images         | 624           |
| Normalization       | 1./255        |

## Transfer Learning Models

Three pretrained CNN architectures are evaluated:

1. **ResNet50**
2. **DenseNet121**
3. **EfficientNetB0**

All models use ImageNet pretrained weights with the classification head replaced for the binary pneumonia classification task.

### Model Parameter Comparison

| **Model**      | **Total Parameters** |
| -------------- | -------------------: |
| ResNet50       |           24,120,705 |
| DenseNet121    |            7,304,257 |
| EfficientNetB0 |            4,382,884 |

## Model Training

Each model is trained for up to:

```text
10 epochs
```

The optimizer used is:

```python
Adam(learning_rate=1e-4)
```

The loss function is:

```text
Binary Crossentropy
```

The following metrics are monitored during training:

* Accuracy
* AUC
* Precision
* Recall

### Training Callbacks

The project uses:

* **EarlyStopping**
* **ReduceLROnPlateau**
* **ModelCheckpoint**
* **TensorBoard**

The best model checkpoint is saved according to validation loss.

## Model Evaluation

The trained models are evaluated using the test dataset.

The evaluation pipeline generates:

* Accuracy
* Precision
* Recall
* F1-score
* AUC
* Confusion Matrix
* ROC Curve
* Classification Report

The classification metrics are calculated separately for the PNEUMONIA class, while accuracy and ROC-AUC are also used for overall model comparison.

## Model Comparison

A comparison table is generated for all three models using:

```text
Accuracy
Precision
Recall
F1-score
AUC
```

The model with the highest AUC is selected as the best-performing model for explainability and subsequent prediction.

### Models Compared

| **Model**      | **Evaluation**                       | **Explainability** |
| -------------- | ------------------------------------ | ------------------ |
| ResNet50       | Accuracy, Precision, Recall, F1, AUC | Grad-CAM           |
| DenseNet121    | Accuracy, Precision, Recall, F1, AUC | Grad-CAM           |
| EfficientNetB0 | Accuracy, Precision, Recall, F1, AUC | Grad-CAM           |

## Explainable AI — Grad-CAM & Grad-CAM++

To improve interpretability, the project implements **Grad-CAM and Grad-CAM++**.

The generated heatmaps highlight the regions of the chest X-ray that contribute to the model's prediction.

The visualization includes:

```text
Original X-ray
      ↓
Grad-CAM Heatmap
      ↓
Heatmap Overlay
```

The notebook selects the model with the highest AUC for Grad-CAM visualization. In the recorded experiment, **DenseNet121** was selected as the best-performing model for Grad-CAM explainability.

Grad-CAM demonstrations are generated for both:

* NORMAL chest X-ray
* PNEUMONIA chest X-ray

## Interactive Prediction

The notebook contains an interactive image-upload interface.

A user can upload a custom chest X-ray image and receive:

* Predicted class
* Confidence score
* Original uploaded image
* Grad-CAM heatmap
* Heatmap overlay

The prediction interface supports:

```text
Upload X-ray
      ↓
Image Preprocessing
      ↓
Best Model Prediction
      ↓
NORMAL / PNEUMONIA
      ↓
Confidence Score
      ↓
Grad-CAM Visualization
```

## Final Model Artifacts

The notebook exports the best-performing model in multiple formats:

```text
best_model.h5
best_model.keras
SavedModel/
```

The native Keras format is also generated for compatibility with modern Keras workflows.

> Large model artifacts are optional for the GitHub submission. The notebook contains the complete code required to recreate and export the models.

## Project Structure

```text
Pneumonia-Detection/
│
├── README.md
├── requirements.txt
│
├── Pneumonia_Detection_Chest_Xray.ipynb
│
├── src/
│   └── README.md
│
├── docs/
│   └── README.md
│
└── results/
    └── README.md
```

The complete dataset should **not** be committed to the repository because it is automatically downloaded through KaggleHub during notebook execution.

Generated model files such as `.h5`, `.keras`, and `SavedModel/` may also be kept outside the repository if their size makes GitHub upload impractical.

## Requirements

The project uses Python and the following major libraries:

```text
tensorflow
keras
kagglehub
opencv-python
matplotlib
seaborn
scikit-learn
pandas
numpy
Pillow
ipywidgets
shap
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

Or install them directly:

```bash
pip install tensorflow kagglehub opencv-python matplotlib seaborn scikit-learn pandas numpy pillow ipywidgets shap
```

The recorded Colab environment used TensorFlow `2.20.0`.


## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow
* KaggleHub
* IPyWidgets
* Transfer Learning
* ResNet50
* DenseNet121
* EfficientNetB0
* Grad-CAM
* Grad-CAM++

## Limitations

* The project performs binary classification only.
* The validation split contains a relatively small number of images in the recorded dataset run.
* Model performance depends on the quality and distribution of the training dataset.
* The system is intended as a machine-learning research/project implementation and not as a replacement for professional medical diagnosis.
* Grad-CAM visualizations provide model interpretability but do not constitute clinical evidence.
* The models require significant computational resources for training.
* The project currently focuses on chest X-ray image information and does not incorporate additional patient clinical data.

## Future Scope

* Evaluate the system on larger and more diverse datasets.
* Improve validation methodology and dataset balancing.
* Perform systematic hyperparameter optimization.
* Explore additional state-of-the-art CNN and vision transformer architectures.
* Improve Grad-CAM++ visualization and explainability analysis.
* Develop a standalone web-based prediction interface.
* Deploy the trained model as a healthcare AI demonstration application.
* Add patient metadata and clinical features where appropriate.
* Perform external validation on independent datasets.
* Investigate model calibration and uncertainty estimation.

## Author

**Isha Agarwal**

B.Tech Computer Science and Engineering (CSE)
Institute of Engineering & Management, Kolkata (UEM Kolkata)
IEEE SMC Student Branch Chapter, KGEC
Research Internship Programme 2026

**Group:** 3
**Project Type:** Individual Project