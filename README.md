# DeepHAR-MLOps: Smartphone Human Activity Recognition

**Course:** CMPE 258 Deep Learning  
**Team:** Vineeth, Siddharth, Arshan, Soham  
**Task:** Classify human activities from smartphone accelerometer and gyroscope time-series windows.

This repository contains an end-to-end deep learning and MLOps-style pipeline for Human Activity Recognition (HAR) from smartphone inertial measurement unit (IMU) data. The model receives a 2.56 second window of raw sensor signals and predicts one of six activities:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

The project demonstrates that architecture matters for time-series sensor learning. A flattened MLP baseline has no explicit temporal inductive bias, while 1D CNN, CNN-LSTM, TCN, and attention-based sequence models are designed to exploit local and sequential movement patterns.

## Why this matters

Smartphone motion sensors are low-power and always available. A reliable HAR system can support fitness tracking, fall detection, elderly care, context-aware mobile apps, and health monitoring without requiring GPS, camera input, or manual user labels.

## Current baseline results from the original notebook

These are the baseline results produced in `notebooks/HAR_Pipeline.ipynb` using the UCI HAR subject-based train/test split.

| Model | Test accuracy | Macro F1 | Train time |
|---|---:|---:|---:|
| MLP | 91.38% | 0.9135 | 82s |
| CNN | 91.79% | 0.9189 | 529s |
| CNN-LSTM | 93.25% | 0.9335 | 829s |

The strongest baseline is CNN-LSTM, supporting the central hypothesis that sequence-aware models improve cross-subject activity recognition.

## Repository highlights

```text
.
├── app/                     # Gradio inference UI
├── configs/                 # YAML experiment configs
├── docs/                    # Dataset, model card, MLOps notes, checklist
├── notebooks/               # Original Colab notebook
├── reports/                 # Final report draft and proposal
├── scripts/                 # Convenience shell commands
├── slides/                  # Slide deck and outline
├── src/deep_har/            # Reusable training/inference package
├── tests/                   # Unit tests for reproducibility helpers
├── artifacts/               # Baseline metrics and generated plots
├── .github/workflows/       # CI checks
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Quick start

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Train a model

The dataset is downloaded automatically from the UCI Machine Learning Repository.

```bash
python -m deep_har.train --model cnn_lstm --epochs 40 --batch-size 64
```

For a quick smoke test:

```bash
python -m deep_har.train --model cnn --epochs 2 --quick-run
```

### 3. Evaluate a checkpoint

```bash
python -m deep_har.evaluate \
  --model-path outputs/checkpoints/cnn_lstm_best.keras \
  --normalizer-path outputs/preprocessing/normalizer.npz
```

### 4. Run inference on a 128 x 9 CSV window

```bash
python -m deep_har.infer \
  --model-path outputs/checkpoints/cnn_lstm_best.keras \
  --normalizer-path outputs/preprocessing/normalizer.npz \
  --input examples/sample_window.csv
```

### 5. Launch the Gradio demo

```bash
python app.py
```

The UI supports sample windows and uploaded CSV/NPY files. It returns the predicted activity, class probabilities, and a sensor-signal plot.

## Experiment plan

Required comparisons:

1. **Architecture comparison:** MLP vs CNN vs CNN-LSTM vs TCN vs CNN-BiLSTM-Attention.
2. **Sensor ablation:** body accelerometer only, gyroscope only, total accelerometer only, all sensors.
3. **Window ablation:** compare 64, 128, and 256 timestep windows if time permits.
4. **Augmentation ablation:** no augmentation vs jitter/scaling/time masking.
5. **Training sweep:** learning rate, dropout, batch size, hidden units.

Run the provided ablation command:

```bash
python -m deep_har.ablations --epochs 15 --models mlp cnn cnn_lstm tcn attention
```

## MLOps components

This repo includes:

- Reproducible training script with seeded runs
- Dataset download and preprocessing modules
- Saved normalizer and model checkpoints
- MLflow-compatible logging hooks
- TensorBoard callback support
- Gradio inference application
- Dockerfile for containerized demo/training
- GitHub Actions smoke tests
- Report, slides, model card, data card, and submission checklist

## Team contributions

| Team member | Primary ownership | Deliverables |
|---|---|---|
| Vineeth | Model architecture and training pipeline | MLP/CNN/CNN-LSTM/TCN/attention models, training script, baseline metrics |
| Siddharth | MLOps and reproducibility | MLflow hooks, Dockerfile, CI workflow, repo organization |
| Arshan | Demo and inference UX | Gradio app, upload/sample inference, confidence visualization |
| Soham | Experiments and documentation | Ablation studies, plots, report, slides, error analysis |

## Submission checklist

Before submitting, complete the items in `docs/submission_checklist.md`. At minimum, push this repo to a public GitHub repository and include:

- README with project summary and links
- Source code
- Notebook
- Report
- Slide deck
- Screenshots of the app and training metrics
- Demo video link
- Model artifacts or download instructions
- Clear team contribution section

## Notes for graders

The original Colab notebook is preserved in `notebooks/HAR_Pipeline.ipynb`. The production-style version is implemented as a reusable Python package in `src/deep_har` with training, evaluation, inference, ablation, and web demo entry points.
