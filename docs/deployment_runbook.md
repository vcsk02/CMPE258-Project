# Deployment Runbook

## Local deployment

```bash
make setup
make train-quick
python app.py
```

Open the Gradio URL, upload `examples/sample_window.csv`, and click **Predict activity**.

## Docker deployment

```bash
docker build -t deep-har-mlops .
docker run -p 7860:7860 deep-har-mlops
```

For a real deployment, mount trained outputs into the container:

```bash
docker run -p 7860:7860 -v "$PWD/outputs:/app/outputs" deep-har-mlops
```

## Hugging Face Spaces deployment

1. Create a Gradio Space.
2. Push `app.py`, `app/`, `src/`, `requirements.txt`, `pyproject.toml`, `examples/`, and trained `outputs/` artifacts.
3. Set the Space SDK to Gradio.
4. Verify sample inference and add the public Space link to README.

## Cloud MLOps deployment outline

The same pipeline can be moved to SageMaker, Vertex AI, Azure ML, or Databricks:

1. Package source code and `configs/train_config.yaml`.
2. Run `python -m deep_har.train` as the managed training job.
3. Store `.keras`, `normalizer.npz`, `reference_profile.json`, metrics, and plots in cloud artifact storage.
4. Gate model promotion on macro F1 and latency thresholds.
5. Deploy the selected checkpoint behind Gradio/API.
6. Log confidence, latency, class distribution, and drift score.
7. Trigger retraining when drift or performance gates fail.

## Promotion gates

Suggested minimum gates for the final demo:

- macro F1 >= baseline CNN-LSTM result or explain why not
- no single class F1 collapse below 0.80
- average inference latency captured in metrics JSON
- Gradio sample inference works from a clean checkout
- tests pass in GitHub Actions


## Clean-clone inference smoke test

```bash
python -m deep_har.infer --model-path outputs/checkpoints/demo_centroid_model.npz --normalizer-path outputs/preprocessing/normalizer.npz --input examples/sample_window.csv --explain
python app.py
```

For the final deep model, replace the `.npz` path with `outputs/checkpoints/sensor_transformer_best.keras` after training.
