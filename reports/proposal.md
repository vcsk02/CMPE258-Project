# Proposal: DeepHAR-MLOps with Sensor Transformer and Prompted Explanations

## Title

DeepHAR-MLOps: Foundation-Model-Inspired Smartphone Human Activity Recognition with Prompt-Engineered Explanations

## Authors

Vineeth Chandra Sai Kandukuri, Siddharth Rao Kartik, Arshan Bhanage, Soham Jain

## Abstract

This project builds an end-to-end MLOps-compliant deep learning system for smartphone human activity recognition. The input is a 2.56-second accelerometer/gyroscope window and the output is one of six activities: walking, walking upstairs, walking downstairs, sitting, standing, or laying. We compare classical and sequence-aware neural architectures including MLP, 1D CNN, CNN-LSTM, TCN, CNN-BiLSTM-Attention, and a compact sensor-transformer. To connect the project to foundation models and prompt engineering, the application includes a prompt-engineered explanation layer that converts model probabilities and sensor statistics into a controlled LLM-ready explanation artifact. The final deliverable is not a Colab-only submission: it includes reusable training code, inference code, Gradio app, Dockerfile, CI workflow, MLOps documentation, ablation tooling, final report, slides, screenshots, and demo recordings.

## Problem

Smartphones already contain accelerometers and gyroscopes that capture motion over time. A robust HAR model can support fitness, health monitoring, fall-risk research, and context-aware mobile apps without GPS, cameras, microphones, or manual user labels. The modeling challenge is temporal: activities are defined by patterns across time, not isolated numeric samples.

## Dataset

We use the UCI Human Activity Recognition dataset. Each example is a 128 timestep by 9 channel smartphone IMU window sampled at 50 Hz. The project uses the original subject-based train/test split so evaluation occurs on people not seen during training.

## Methods

We will compare:

1. MLP baseline
2. 1D CNN
3. CNN-LSTM
4. Temporal Convolutional Network
5. CNN-BiLSTM-Attention
6. Sensor Transformer

The pipeline includes per-channel training-only normalization, optional jitter/scaling/time-mask augmentation, early stopping, checkpointing, TensorBoard logging, optional MLflow logging, and saved metrics/plots.

## LLM / foundation-model component

The project includes a `sensor_transformer` architecture and a prompt-engineered explanation layer. The explanation layer packages prediction context into a controlled system/user prompt with probabilities, confidence, drift status, and sensor statistics. This prompt can be sent to an LLM in a cloud deployment, while the deterministic fallback keeps the class demo reproducible without external API keys.

## Experiments

Planned experiments:

- Architecture comparison across all supported models.
- Sensor ablation: all channels, body acceleration, body gyroscope, total acceleration.
- Augmentation ablation: no augmentation vs jitter/scaling/time masking.
- Hyperparameter sweep: learning rate and batch size.
- Error analysis using confusion matrices and per-class F1.

## Expected deliverables

- Public GitHub repository with a complete README
- Source code under `src/deep_har/`
- Gradio inference website
- Training/evaluation/inference CLIs
- Dockerfile and GitHub Actions workflow
- MLflow/TensorBoard artifacts
- Data card, model card, MLOps plan, DeepWiki/Repomix plan
- Final report and slide deck
- Screenshots of app/training metrics
- Short demo video and long presentation recording
- Team contribution table
