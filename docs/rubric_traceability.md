# Rubric Traceability Matrix

| Rubric item | Evidence in repository | Final action before submit |
|---|---|---|
| Team size <= 4 | Four names in README/report | Confirm exact names and roles |
| Public GitHub URL | README has placeholder | Push repo public and paste URL into spreadsheet |
| MUST HAVE README | `README.md` rewritten with setup, training, inference, deliverables, contributions | Replace placeholder links |
| All artifacts included | `artifacts/`, `reports/`, `slides/`, `docs/`, `notebooks/`, `src/` | Add final screenshots/videos/results |
| Gradio UX minimum | `app.py`, `app/gradio_app.py` | Add screenshot to `artifacts/screenshots/` |
| Model training pipeline | `src/deep_har/train.py` | Run final model and save outputs |
| Inference pipeline | `src/deep_har/infer.py`, Gradio app | Demo with sample CSV |
| Complex model | CNN-LSTM, TCN, attention, sensor-transformer | Train and compare advanced models |
| Dataset and preprocessing | `docs/data_card.md`, `src/deep_har/data.py`, `src/deep_har/preprocess.py` | Include dataset stats in final report |
| Metrics and visualizations | `src/deep_har/visualization.py`, TensorBoard, MLflow, plots | Save final plots/screenshots |
| Ablation studies | `src/deep_har/ablations.py` | Run architecture/sensor/augmentation ablations |
| Sweeps | `src/deep_har/sweeps.py` | Run small LR/batch sweep |
| LLM/foundation/prompting | `src/deep_har/prompting.py`, `docs/llm_prompting.md`, `sensor_transformer` | Show prompt output in demo |
| MLOps practices | Dockerfile, CI, DVC-style `dvc.yaml`, MLflow/TensorBoard hooks | Add cloud screenshots if using SageMaker/Vertex/Databricks/HF Spaces |
| DeepWiki/Repomix | `docs/deepwiki_repomix.md` | Run the chosen documentation tool and link output |
| Team contributions | README and `docs/team_contributions.md` | Add video timestamp ownership |
| Final report sections | `reports/final_report.md` | Export to PDF if required |
| Slides | `slides/` | Update PPT from outline if needed |
| No RL | Project uses supervised classification only | State this in report/demo |
| No Colab-only submission | Reusable package, scripts, Docker, CI | Keep notebook as supplementary only |
