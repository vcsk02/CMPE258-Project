# CMPE Submission Checklist

## GitHub repository

- [ ] Public GitHub repository is accessible without login.
- [ ] README explains the project, setup, training, inference, MLOps, results, and deliverables.
- [ ] Team member contributions are clearly listed.
- [ ] Original notebook is included only as supplementary material.
- [ ] Production-style source code is included.
- [ ] Requirements file is included.
- [ ] Dockerfile is included.
- [ ] Gradio app runs locally.
- [ ] GitHub link is pasted in the course spreadsheet.

## Model and training

- [ ] At least three models are trained and compared.
- [ ] One advanced sequence model is trained: TCN, attention, or sensor-transformer.
- [ ] Test metrics are generated on unseen subjects.
- [ ] Confusion matrices are saved.
- [ ] Per-class F1 scores are included.
- [ ] Training curves are included.
- [ ] Inference latency is reported.
- [ ] Model parameter count is reported.

## LLM / foundation / prompting

- [ ] Sensor-transformer experiment is included or clearly discussed.
- [ ] Prompt-engineered explanation output is shown in Gradio.
- [ ] Prompting design is documented in `docs/llm_prompting.md`.
- [ ] Demo explains that the LLM layer summarizes model outputs and uncertainty, while the trained classifier performs prediction.

## Experiments

- [ ] Architecture comparison completed.
- [ ] Sensor ablation completed.
- [ ] Augmentation ablation completed.
- [ ] Hyperparameter sweep or small grid search completed.
- [ ] Error analysis included.
- [ ] At least 20% of presentation/report focuses on visualization, metrics, ablation studies, and sweeps.

## MLOps

- [ ] TensorBoard screenshot added to `artifacts/screenshots/`.
- [ ] MLflow screenshot added if `--use-mlflow` is used.
- [ ] CI screenshot added.
- [ ] Docker build/run tested.
- [ ] Artifact manifest regenerated.
- [ ] DeepWiki or Repomix output linked.
- [ ] Optional cloud deployment screenshots added if pursuing extra credit.

## App/demo

- [ ] Gradio app screenshot added to `artifacts/screenshots/`.
- [ ] Inference demo video recorded.
- [ ] Demo video link added to README and `artifacts/demo_video_link.md`.
- [ ] Long team presentation link added to `artifacts/presentation_recording_link.md`.
- [ ] Inference works on `examples/sample_window.csv`.

## Final deliverables

- [ ] Final report exported to PDF if required.
- [ ] Slide deck reviewed and updated.
- [ ] All metrics and screenshots included.
- [ ] Public GitHub link submitted to the course spreadsheet.
