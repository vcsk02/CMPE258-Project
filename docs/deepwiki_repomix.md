# DeepWiki / Repomix Documentation Plan

The rubric asks the team to leverage DeepWiki or Repomix to document the codebase with artifacts. This repository is ready for either workflow.

## Option A: Repomix

Install and run from the repository root:

```bash
npx repomix --output artifacts/repomix-output.md --style markdown
```

Recommended include/exclude behavior:

- Include: `README.md`, `src/`, `app/`, `docs/`, `reports/`, `configs/`, `scripts/`, `tests/`, `dvc.yaml`, `Dockerfile`, `.github/workflows/ci.yml`
- Exclude: `.git/`, `.venv/`, `data/raw/`, large model checkpoints, TensorBoard event files, generated caches

After running, link `artifacts/repomix-output.md` in the final README.

## Option B: DeepWiki

1. Push the repository to a public GitHub URL.
2. Open it in DeepWiki.
3. Verify that DeepWiki indexes source files, docs, and MLOps artifacts.
4. Add the DeepWiki URL to the README deliverables table.
5. Capture a screenshot and place it in `artifacts/screenshots/`.

## Suggested documentation questions to ask the tool

- Explain the training pipeline from dataset download to checkpoint saving.
- Explain the inference path from Gradio upload to class probabilities.
- Summarize the model architectures and why the sensor-transformer was added.
- Identify where ablation studies and sweeps are implemented.
- Identify the MLOps maturity level and missing pieces for level 4 automation.
