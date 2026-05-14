# Team Contributions

Use this file in the README, report, slides, and video presentation so the grader can clearly see what each person did beyond copying existing code or relying on a notebook.

## Vineeth Chandra Sai Kandukuri

**Primary role:** model architecture and training pipeline.

Contributions:

- Converted the HAR experiment into reusable package entry points.
- Owned model comparison among MLP, CNN, CNN-LSTM, TCN, attention, and sensor-transformer.
- Defined the training/evaluation protocol using subject-based train/test split.
- Interpreted accuracy, macro F1, confusion matrix, and per-class failure modes.
- Prepared model-selection rationale for the final report and presentation.

## Siddharth Rao Kartik

**Primary role:** MLOps, reproducibility, and deployment packaging.

Contributions:

- Organized the repo for reproducible training and inference.
- Added Dockerfile, Makefile, GitHub Actions, and DVC-style pipeline metadata.
- Added MLflow/TensorBoard tracking flow and artifact-management expectations.
- Documented cloud deployment paths for Hugging Face Spaces, Vertex AI, SageMaker, Azure ML, or Databricks.
- Prepared MLOps maturity discussion and monitoring/retraining plan.

## Arshan Bhanage

**Primary role:** inference UX and demo workflow.

Contributions:

- Built the Gradio upload/sample inference experience.
- Added probability table and sensor-window visualization to make predictions explainable.
- Integrated prompt-engineered explanation output and monitoring status into the UX.
- Prepared the demo flow for showing model inference on a real CSV window.
- Owned screenshots and short video demo artifacts.

## Soham Jain

**Primary role:** experiments, visualizations, and documentation.

Contributions:

- Owned ablation plan for model architecture, sensor subset, augmentation, and hyperparameters.
- Prepared visualizations: training curves, confusion matrices, and comparison plots.
- Drafted final report sections and slide narrative.
- Documented data card, model card, submission checklist, and rubric traceability.
- Prepared error analysis and future-work recommendations.
