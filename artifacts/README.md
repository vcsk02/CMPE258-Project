# Artifacts

This directory stores submission artifacts such as metrics, plots, screenshots, generated manifests, and video links.

Expected final contents:

- `baseline_results/`: metrics and plots from the original notebook
- `screenshots/`: Gradio app, TensorBoard, MLflow, GitHub repo, CI, and optional cloud deployment screenshots
- `artifact_manifest.md`: generated audit trail from `python scripts/generate_artifact_manifest.py`
- `demo_video_link.md`: short demo video link
- `presentation_recording_link.md`: long full-team presentation link
- optional `repomix-output.md`: generated codebase documentation if using Repomix

Do not commit huge raw datasets or large model checkpoints unless GitHub storage limits allow it. If checkpoints are too large, include reproducible training commands and cloud/download links.
