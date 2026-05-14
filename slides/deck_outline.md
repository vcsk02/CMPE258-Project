# Slide Deck Outline

## Slide 1: Title

DeepHAR-MLOps: Foundation-Model-Inspired Human Activity Recognition  
Vineeth, Siddharth, Arshan, Soham

## Slide 2: Motivation

- Smartphones already collect motion signals.
- HAR supports fitness, fall-risk research, elderly care, and context-aware apps.
- No GPS, camera, microphone, or manual input is required.

## Slide 3: Problem Statement

Input: 2.56-second sensor window, 128 timesteps x 9 channels.  
Output: one of six activities.  
Goal: generalize to unseen subjects.

## Slide 4: Dataset

- UCI HAR dataset
- 30 volunteers
- Accelerometer and gyroscope
- Subject-based split
- Training-only normalization

## Slide 5: End-to-End Pipeline

Download data -> split/normalize -> train -> checkpoint -> evaluate -> log metrics -> Gradio inference -> prompt explanation -> drift monitoring.

## Slide 6: Models

- MLP baseline
- 1D CNN
- CNN-LSTM
- TCN
- CNN-BiLSTM-Attention
- Sensor Transformer

## Slide 7: Foundation Model / Prompting Connection

- Sensor-transformer uses self-attention and positional embeddings.
- Prompt layer converts probabilities + sensor stats into an LLM-ready explanation.
- Deterministic fallback keeps the demo reproducible.

## Slide 8: Baseline Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| MLP | 91.38% | 0.9135 |
| CNN | 91.79% | 0.9189 |
| CNN-LSTM | 93.25% | 0.9335 |

## Slide 9: Ablations and Sweeps

Show final tables/plots for:

- architecture comparison
- sensor subsets
- augmentation
- learning rate / batch size

## Slide 10: Error Analysis

Most common confusion: SITTING vs STANDING.  
Walking classes are easier due to periodic motion.  
Use confusion matrix and per-class F1.

## Slide 11: MLOps Artifacts

- Reusable scripts
- Saved preprocessing artifact
- Reference drift profile
- TensorBoard/MLflow hooks
- Dockerfile
- CI tests
- DVC-style pipeline
- Artifact manifest

## Slide 12: Gradio Demo

Show:

- file upload / sample window
- prediction
- probabilities
- sensor plot
- prompt-engineered explanation
- monitoring status

## Slide 13: Team Contributions

- Vineeth: models and training
- Siddharth: MLOps and deployment
- Arshan: demo and inference UX
- Soham: experiments, plots, report, slides

## Slide 14: Conclusion and Future Work

- Sequence-aware models improve HAR performance.
- The repo is structured as an MLOps project, not a Colab-only demo.
- Future work: cloud pipeline, drift-triggered retraining, real-time mobile inference, phone-placement robustness.
