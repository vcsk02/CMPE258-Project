# LLM and Prompt-Engineering Component

The core predictive model is a trained time-series classifier. To satisfy the project requirement that the selected project connect to large language models, foundation models, prompting, or prompt engineering, the revised system adds two pieces:

1. **Sensor-transformer model:** `src/deep_har/models.py` includes `sensor_transformer`, a compact transformer encoder for raw IMU windows. It uses learned sensor-token projection, learned positional embeddings, multi-head self-attention, feed-forward blocks, layer normalization, residual connections, and a classification head.
2. **Prompt-engineered explanation layer:** `src/deep_har/prompting.py` converts structured model outputs into a controlled LLM-ready prompt. It includes a system prompt, a user prompt containing probabilities and sensor statistics, and a deterministic fallback explanation so the demo works without an API key.

## Why this is useful

The classifier answers: “Which activity is this window most likely to represent?”

The prompt layer answers: “How should we explain this prediction responsibly to a project-demo audience?”

This is a practical foundation-model pattern: use a specialized trained model for perception/prediction, then use a prompt-controlled language model layer for explanation, caveats, and operator-facing summaries.

## Prompt safety controls

The prompt instructs the explanation assistant to:

- avoid medical certainty
- mention uncertainty and likely failure modes
- keep the explanation concise
- include what data was used
- include a caveat and a reliability-improvement step

## Demo behavior

The Gradio app displays both:

- a deterministic explanation, which is always runnable
- the exact system/user prompt artifact, which can be pasted into or wired to an LLM endpoint

This keeps the project reproducible for grading while still demonstrating prompt engineering.
