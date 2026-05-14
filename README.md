# DeepHAR-MLOps: Foundation-Model-Inspired Human Activity Recognition

**Course:** CMPE 257 / Deep Learning MLOps Project  
**Public GitHub:** https://github.com/vcsk02/CMPE258-Project  
**Team:** Vineeth Chandra Sai Kandukuri, Siddharth Rao Kartik, Arshan Bhanage, Soham Jain  
**Primary task:** classify smartphone accelerometer/gyroscope windows into human activities.  
**LLM/foundation-model component:** a compact sensor-transformer architecture plus a prompt-engineered explanation layer that converts model outputs, probabilities, and drift statistics into a controlled LLM-ready explanation artifact.

This repository is a submission-ready, non-Colab project archive for an end-to-end machine learning system. It contains reusable source code, model training and inference pipelines, ablation/sweep tooling, MLOps artifacts, documentation, report drafts, slide artifacts, and a Gradio UX.

## Problem summary

The model receives a **2.56 second, 128 timestep, 9-channel** smartphone IMU window and predicts one of six activities:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

This matters because robust activity recognition can support fitness tracking, fall-risk research, elderly-care applications, context-aware mobile apps, and health monitoring without camera, microphone, GPS, or manual labels.

## What changed for the rubric

The original project already had HAR models and a Gradio app. The revised version now more directly addresses the rubric:

| Rubric requirement | Repository evidence |
|---|---|
| Public GitHub with all artifacts | `README.md`, `artifacts/`, `reports/`, `slides/`, `docs/`, `src/`, `tests/` |
| Not just Colab | Production package under `src/deep_har/`, CLI scripts, Dockerfile, CI, DVC-style pipeline |
| Model training component | `python -m deep_har.train` trains MLP/CNN/CNN-LSTM/TCN/attention/sensor-transformer models |
| Inference website | `app.py` launches Gradio for upload/sample prediction |
| Foundation model / prompting relevance | `sensor_transformer` model and `src/deep_har/prompting.py` prompt-engineered explanation artifact |
| Ablations and sweeps | `src/deep_har/ablations.py`, `src/deep_har/sweeps.py` |
| Visualization and metrics | confusion matrices, training curves, TensorBoard logs, MLflow hooks, artifact manifest |
| MLOps maturity | Docker, GitHub Actions, saved normalizer, checkpointing, reference drift profile, model cards/data cards |
| Team contributions | README section plus `docs/team_contributions.md` |
| DeepWiki/Repomix documentation | `docs/deepwiki_repomix.md` |

## Repository layout

```text
.
├── app/                         # Gradio inference UX
├── artifacts/                   # Results, screenshots, video links, manifest
├── configs/                     # Training config and MLOps thresholds
├── data/                        # Dataset instructions; raw data is downloaded automatically
├── docs/                        # Data card, model card, MLOps plan, rubric traceability
├── notebooks/                   # Original notebook kept as supplementary material
├── reports/                     # Proposal and final report draft
├── scripts/                     # Convenience commands and artifact manifest generator
├── slides/                      # Slide deck and outline
├── src/deep_har/                # Reusable ML package
├── tests/                       # Unit tests
├── .github/workflows/ci.yml     # CI smoke tests
├── Dockerfile
├── Makefile
├── dvc.yaml
└── README.md
```

## Current baseline and packaged artifacts

Baseline results from the original notebook using the UCI HAR subject-based train/test split:

| Model | Test accuracy | Macro F1 | Train time |
|---|---:|---:|---:|
| MLP | 91.38% | 0.9135 | 82s |
| CNN | 91.79% | 0.9189 | 529s |
| CNN-LSTM | 93.25% | 0.9335 | 829s |

The strongest executed baseline is CNN-LSTM, supporting the hypothesis that sequence-aware models improve cross-subject activity recognition. The repo also implements TCN, CNN-BiLSTM-Attention, and a compact `sensor_transformer` for the final TensorFlow comparison.

This archive now includes clean-download inference artifacts:

- `outputs/checkpoints/demo_centroid_model.npz`
- `outputs/preprocessing/normalizer.npz`
- `outputs/preprocessing/reference_profile.json`
- `outputs/results/demo_centroid_test_metrics.json`
- `outputs/plots/demo_centroid_confusion_matrix.png`

The packaged `.npz` checkpoint is a lightweight smoke-test/demo model so the CLI and Gradio app work immediately. The scored deep-learning path remains `python -m deep_har.train`, which generates `.keras` checkpoints for CNN-LSTM, TCN, attention, and sensor-transformer models.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Train the model

The dataset is downloaded automatically from the UCI Machine Learning Repository.

```bash
python -m deep_har.train --model sensor_transformer --epochs 40 --batch-size 64 --use-mlflow
```

For a fast smoke test:

```bash
python -m deep_har.train --model cnn --epochs 2 --quick-run
```

Training creates:

- `outputs/checkpoints/<model>_best.keras`
- `outputs/preprocessing/normalizer.npz`
- `outputs/preprocessing/reference_profile.json`
- `outputs/results/<model>_test_metrics.json`
- `outputs/results/<model>_classification_report.json`
- `outputs/results/<model>_model_summary.txt`
- `outputs/plots/<model>_training_curves.png`
- `outputs/plots/<model>_confusion_matrix.png`
- `outputs/tensorboard/<model>/`

## Evaluate a checkpoint

```bash
python -m deep_har.evaluate \
  --model-path outputs/checkpoints/sensor_transformer_best.keras \
  --normalizer-path outputs/preprocessing/normalizer.npz
```

## Run CLI inference

```bash
python -m deep_har.infer \
  --model-path outputs/checkpoints/demo_centroid_model.npz \
  --normalizer-path outputs/preprocessing/normalizer.npz \
  --input examples/sample_window.csv \
  --explain
```

The `--explain` flag returns the prediction, probabilities, deterministic fallback explanation, and an LLM-ready system/user prompt artifact.

## Launch the Gradio website

```bash
python app.py
```

The UX supports a sample window or uploaded CSV/NPY file. It returns:

- predicted activity
- confidence
- class probabilities
- sensor-signal plot
- prompt-engineered explanation
- LLM prompt artifact
- drift-monitoring status

## Completed and reproducible experiment artifacts

Executed notebook baselines and generated repo artifacts are saved under `outputs/` and `artifacts/`:

| Artifact | Location |
|---|---|
| UCI HAR baseline table | `outputs/results/uci_notebook_baseline_results.csv` |
| Completed architecture ablation summary | `outputs/results/ablation_summary.csv` |
| Sweep configuration/results seed table | `outputs/results/hyperparameter_sweep_results.csv` |
| Packaged inference checkpoint metrics | `outputs/results/demo_centroid_test_metrics.json` |
| Confusion matrices / plots | `outputs/plots/` and `artifacts/screenshots/` |
| Report PDF | `reports/final_report.pdf` |

Commands for the final TensorFlow deep-model runs:

```bash
python -m deep_har.ablations --epochs 15 --models mlp cnn cnn_lstm tcn attention sensor_transformer
python -m deep_har.sweeps --epochs 10 --models cnn cnn_lstm sensor_transformer
tensorboard --logdir outputs/tensorboard
python scripts/generate_artifact_manifest.py
```

## MLOps design

This project targets **MLOps level 2** by default and includes pieces needed to discuss level 3/4 as future or optional cloud work.

Implemented:

- automated training script
- saved preprocessing normalizer
- saved model checkpoint
- reproducible seed control
- centralized metrics and plots
- TensorBoard callback
- optional MLflow logging with `--use-mlflow`
- CLI evaluation and inference
- Gradio web inference
- Docker container
- GitHub Actions test workflow
- drift-reference profile for production monitoring
- artifact manifest generator

Optional extra-credit deployment paths:

- Hugging Face Spaces: use `app.py` as entry point.
- Vertex AI / SageMaker / Azure ML / Databricks: wrap `python -m deep_har.train` as the training job and deploy the saved `.keras` checkpoint behind the Gradio/API service.
- Level 4 extension: schedule drift checks, trigger retraining on high drift, and auto-promote only if validation/test gates pass.

## Deliverables checklist and links

| Deliverable | Location/status |
|---|---|
| Public GitHub URL | https://github.com/vcsk02/CMPE258-Project |
| README | `README.md` |
| Source code | `src/deep_har/`, `app/`, `scripts/` |
| Gradio app | `app.py`, `app/gradio_app.py` |
| Packaged runnable inference artifacts | `outputs/checkpoints/demo_centroid_model.npz`, `outputs/preprocessing/normalizer.npz` |
| Deep model training pipeline | `src/deep_har/train.py` generates `.keras` checkpoints |
| Notebook | `notebooks/HAR_Pipeline.ipynb` |
| Proposal | `reports/proposal.md` |
| Final report | `reports/final_report.md`, `reports/final_report.pdf` |
| Slide deck | `slides/DeepHAR_MLOps_Deck.pptx` and `slides/deck_outline.md` |
| Screenshots / executed plots | `artifacts/screenshots/` |
| Demo video link file | `artifacts/demo_video_link.md` |
| Long presentation recording file | `artifacts/presentation_recording_link.md` |
| Metrics/results | `artifacts/baseline_results/` and `outputs/results/` |
| DeepWiki/Repomix instructions | `docs/deepwiki_repomix.md` |
| Rubric traceability | `docs/rubric_traceability.md` |

## Team contributions

| Team member | Primary ownership | Non-code/report/demo contribution to highlight |
|---|---|---|
| Vineeth | Model architecture and training pipeline | Defined architecture comparison, training protocol, model-selection rationale, and final metric interpretation |
| Siddharth | MLOps and reproducibility | Owned Docker/CI/MLflow/TensorBoard/DVC-style pipeline and cloud-deployment plan |
| Arshan | Demo and inference UX | Owned Gradio UX, sample-file inference flow, confidence visualization, prompt explanation display, and demo recording |
| Soham | Experiments and documentation | Owned ablations, sweeps, plots, error analysis, slide narrative, and report formatting |

See `docs/team_contributions.md` for the detailed split.

## Final human submission actions

The code archive now includes runnable inference artifacts, plots, results, report files, slides, and screenshots extracted from the executed notebook. The only items that still require the team to add real external links are the short demo video, long team presentation recording, and course spreadsheet submission. Do not invent these links; record the videos, paste the URLs into `artifacts/demo_video_link.md` and `artifacts/presentation_recording_link.md`, then push the repo.