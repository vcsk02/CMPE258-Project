# Rubric Traceability Matrix

| Rubric item | Evidence in repository | Final action before submit |
|---|---|---|
| Team size <= 4 | Four names in README/report | Confirm exact names in Canvas/GitHub submission |
| Public GitHub URL | https://github.com/vcsk02/CMPE258-Project | Paste URL into course spreadsheet |
| MUST HAVE README | `README.md` with setup, training, inference, artifacts, contributions | Keep links current |
| All artifacts included | `artifacts/`, `reports/`, `slides/`, `docs/`, `notebooks/`, `src/`, `outputs/` | Add video URLs after recording |
| Gradio UX minimum | `app.py`, `app/gradio_app.py`, packaged `.npz` inference artifact | Capture live browser screenshot after push/run |
| Model training pipeline | `src/deep_har/train.py`, `dvc.yaml`, Makefile commands | Run final deep model on team machine/cloud for `.keras` checkpoint |
| Inference pipeline | `src/deep_har/infer.py`, Gradio app, `outputs/checkpoints/demo_centroid_model.npz` | Optional: swap default to trained `.keras` for final demo |
| Complex model | CNN-LSTM baseline plus TCN, attention, sensor-transformer implementation | Run sensor-transformer final experiment for stronger score |
| Dataset and preprocessing | `docs/data_card.md`, `src/deep_har/data.py`, `src/deep_har/preprocess.py`, normalizer artifact | Include downloaded UCI data only if allowed/needed |
| Metrics and visualizations | `outputs/results/`, `outputs/plots/`, `artifacts/screenshots/` | Add TensorBoard/MLflow screenshots if generated locally |
| Ablation studies | `outputs/results/ablation_summary.csv`, `src/deep_har/ablations.py` | Complete advanced/sensor/augmentation ablations for 90+ target |
| Sweeps | `outputs/results/hyperparameter_sweep_results.csv`, `src/deep_har/sweeps.py` | Complete final grid if time permits |
| LLM/foundation/prompting | `src/deep_har/prompting.py`, `docs/llm_prompting.md`, sensor-transformer | Show prompt output in recorded demo |
| MLOps practices | Dockerfile, CI, DVC-style `dvc.yaml`, MLflow/TensorBoard hooks, drift profile | Add GitHub Actions screenshot after push |
| DeepWiki/Repomix | `docs/deepwiki_repomix.md` | Generate output after public repo is live |
| Team contributions | README and `docs/team_contributions.md` | Add video timestamps after recording |
| Final report sections | `reports/final_report.md`, `reports/final_report.pdf` | Review final PDF formatting |
| Slides | `slides/DeepHAR_MLOps_Deck.pptx` | Present from this deck or export PDF |
| No RL | Supervised classification only | State in demo/report |
| No Colab-only submission | Reusable package, scripts, Docker, CI; notebook supplementary only | Push full repo, not notebook alone |
