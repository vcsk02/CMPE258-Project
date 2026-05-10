# Proposal: DeepHAR-MLOps

## Title

DeepHAR-MLOps: Sequence-Aware Deep Learning for Smartphone Human Activity Recognition

## Team

Vineeth, Siddharth, Arshan, Soham

## Problem

Smartphones contain accelerometers and gyroscopes that capture how the phone moves over time. We use these sensor readings to recognize human activities without GPS, camera, microphone, or manual input.

## Goal

Train deep learning models that classify a 2.56 second raw sensor window into six activity classes. The main research question is whether sequence-aware models outperform a flattened baseline when evaluated on subjects not seen during training.

## Dataset

UCI HAR smartphone sensor dataset. Each sample is a 128 timestep by 9 channel sensor window.

## Methods

We compare MLP, 1D CNN, CNN-LSTM, TCN, and CNN-BiLSTM-Attention models. The pipeline includes normalization, training, evaluation, ablation studies, and a Gradio inference demo.

## Expected deliverables

- Public GitHub repository
- Training and inference code
- Notebook
- Gradio demo
- MLflow/TensorBoard metrics
- Ablation studies
- Final report
- Slide deck
- Demo video
