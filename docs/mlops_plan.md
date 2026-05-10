# MLOps Plan

## Current maturity target

This repo targets MLOps level 2 style automation for the class project:

- Automated training entry point
- Reproducible preprocessing
- Centralized metrics and artifacts
- Versioned code and configs
- Containerized application
- Web inference demo
- CI smoke tests

## Training pipeline

1. Download UCI HAR dataset.
2. Load train/test subject split.
3. Fit normalizer on training data only.
4. Create train/validation split.
5. Train selected model architecture.
6. Save best checkpoint.
7. Evaluate on unseen test subjects.
8. Save metrics, plots, confusion matrix, and reports.

## Inference pipeline

1. Accept CSV or NPY sensor window.
2. Validate shape.
3. Apply saved training normalizer.
4. Run model prediction.
5. Return top class, confidence, probability table, and signal plot.

## Tracking

The training script can log to MLflow with `--use-mlflow`. TensorBoard logs are saved under `outputs/tensorboard`.

## Deployment options

- Local Gradio app using `python app.py`
- Docker container exposing port 7860
- Hugging Face Spaces by using `app.py` as the Spaces entry point
- Optional cloud MLOps deployment on SageMaker, Vertex AI, Azure ML, or Databricks

## Monitoring extension

For a production extension, log:

- Prediction confidence distribution
- Input signal statistics by channel
- Class distribution drift
- Latency
- Failure cases and user feedback
