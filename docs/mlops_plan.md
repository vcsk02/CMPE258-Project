# MLOps Plan

## Current maturity target

This repo targets **MLOps maturity level 2** for the base submission and documents what is needed for level 3/4 extra credit.

Implemented level 2 evidence:

- Automated training entry point: `python -m deep_har.train`
- Automated evaluation entry point: `python -m deep_har.evaluate`
- Automated inference entry point: `python -m deep_har.infer`
- Gradio web inference UX: `python app.py`
- Reproducible preprocessing with saved `normalizer.npz`
- Saved model checkpoints
- Centralized metrics and artifacts under `outputs/`
- TensorBoard logging
- Optional MLflow logging with `--use-mlflow`
- Dockerfile and Makefile
- GitHub Actions smoke tests
- DVC-style pipeline metadata in `dvc.yaml`
- Drift-reference profile saved during training
- Artifact manifest generator

## Training pipeline

1. Download UCI HAR dataset.
2. Load train/test subject split.
3. Select sensor subset.
4. Create train/validation split with stratification.
5. Fit normalizer on training data only.
6. Save normalizer and drift-reference profile.
7. Optionally augment training windows.
8. Train selected architecture.
9. Save best checkpoint by validation loss.
10. Evaluate on unseen test subjects.
11. Save metrics, plots, predictions, confusion matrix, and model summary.
12. Log to TensorBoard and optionally MLflow.

## Inference pipeline

1. Accept CSV or NPY sensor window.
2. Validate shape.
3. Apply saved training normalizer.
4. Run model prediction.
5. Return top class, confidence, probabilities, and sensor plot.
6. Build prompt-engineered explanation artifact.
7. Compute simple input-drift score if `reference_profile.json` exists.

## Experiment tracking

TensorBoard logs are saved under `outputs/tensorboard`. MLflow can be enabled:

```bash
python -m deep_har.train --model sensor_transformer --epochs 40 --use-mlflow
mlflow ui
```

Minimum tracked metrics:

- accuracy
- macro F1
- train time
- latency per test window
- parameter count
- train/validation/test example counts
- augmentation flag

## Monitoring

The training job saves `outputs/preprocessing/reference_profile.json`. The app uses it to calculate a lightweight input-drift score for each uploaded window.

Suggested production logs:

- timestamp
- model version/checkpoint
- predicted class
- confidence
- probability distribution
- latency
- drift score and level
- upload validation failures
- user feedback if available

## Promotion gates

A model should not be promoted unless:

- macro F1 is at least as strong as the chosen baseline or the tradeoff is justified
- no key class has unacceptable F1 collapse
- inference latency is acceptable for the demo target
- sample inference passes from a clean checkout
- unit tests and CI pass
- data/preprocessing artifacts are versioned with the checkpoint

## Level 3/4 extra-credit path

To claim level 3/4, the team should demo a managed cloud pipeline such as Databricks, Vertex AI, SageMaker, Azure ML, or Hugging Face Spaces plus CI/CD. Required evidence:

- automated train/test pipeline run
- artifact registry or model registry screenshot
- deployment endpoint or public Space URL
- automated evaluation gate
- monitoring dashboard or drift report
- retraining trigger plan or demo
- rollback/promote process
