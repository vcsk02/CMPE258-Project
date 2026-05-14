# DeepHAR-MLOps: Foundation-Model-Inspired Sequence Learning for Smartphone Human Activity Recognition

**Authors:** Vineeth Chandra Sai Kandukuri, Siddharth Rao Kartik, Arshan Bhanage, Soham Jain

## Abstract

Human activity recognition from smartphone inertial sensors enables fitness tracking, fall-risk research, elderly care, and context-aware applications without requiring GPS, cameras, microphones, or manual user labels. This project builds an end-to-end MLOps-style deep learning system for classifying six human activities from 2.56-second accelerometer and gyroscope windows. We compare a flattened MLP baseline with sequence-aware architectures including 1D CNN, CNN-LSTM, Temporal Convolutional Network, CNN-BiLSTM-Attention, and a compact sensor-transformer. The original notebook baseline achieves 91.38% accuracy with MLP, 91.79% with CNN, and 93.25% with CNN-LSTM on the subject-based UCI HAR test split. The revised system adds reproducible training scripts, saved preprocessing artifacts, evaluation reports, ablation and sweep tooling, MLflow/TensorBoard hooks, drift-monitoring reference profiles, a Gradio inference demo, and a prompt-engineered explanation layer that creates an LLM-ready artifact from model probabilities and sensor statistics.

## 1. Introduction

Smartphones contain motion sensors that continuously measure acceleration and angular velocity. These signals can be used to infer whether a user is walking, climbing stairs, sitting, standing, or lying down. Unlike GPS or camera-based approaches, inertial sensing is lower power and less privacy invasive.

The core challenge is that sensor data is sequential. A walking signal is not defined only by individual numeric values; it is defined by periodic and directional changes across time. Therefore, models with temporal inductive bias should perform better than models that flatten the input.

This project studies that question using the UCI HAR dataset and a subject-based evaluation protocol. The final repository is designed as a class-project MLOps package rather than a notebook-only experiment.

## 2. Related Work

Human activity recognition has traditionally used hand-engineered time-domain and frequency-domain features with classical machine learning methods. Deep learning reduces manual feature engineering by learning directly from raw sensor windows. CNNs learn local movement motifs, recurrent models learn longer temporal dependencies, TCNs capture sequence structure through dilated residual convolutions, and attention/transformer layers can model global relationships among timesteps.

Our approach is similar to prior deep HAR systems because it trains on raw IMU windows. It differs from a simple notebook experiment by packaging the work into a reproducible training and inference pipeline with MLOps artifacts, a Gradio demo, CI checks, and prompt-engineered explanations for demo users.

## 3. Data

The project uses the UCI Human Activity Recognition Using Smartphones dataset. Each sample is a 128-timestep window sampled at 50 Hz, corresponding to 2.56 seconds. There are nine sensor channels: body acceleration x/y/z, body gyroscope x/y/z, and total acceleration x/y/z.

The six labels are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING. The original dataset split separates train and test subjects, so the test set contains people not used during training. This is important because it evaluates cross-subject generalization rather than memorization of a person’s motion pattern.

Preprocessing uses per-channel z-score normalization. The mean and standard deviation are fitted on the training data only and reused for validation, test, and inference. The training job also saves a reference profile of training-split channel statistics for drift checks during inference.

## 4. Methods

### 4.1 MLP baseline

The MLP flattens the sensor window and passes it through dense layers with batch normalization and dropout. It is useful as a lower-bound baseline because it can learn nonlinear combinations of input values but has no explicit temporal inductive bias.

### 4.2 1D CNN

The CNN applies temporal convolutions over sensor channels. This lets the model learn local motion motifs such as periodic acceleration patterns during walking.

### 4.3 CNN-LSTM

The CNN-LSTM first extracts local temporal features with convolution layers, then feeds the resulting sequence into an LSTM. This combines local pattern recognition with sequence modeling.

### 4.4 TCN

The Temporal Convolutional Network uses residual dilated convolution blocks. It captures longer temporal context while preserving efficient parallel training.

### 4.5 CNN-BiLSTM-Attention

The attention model uses convolution, bidirectional LSTM, and multi-head self-attention. This provides a stronger sequence-aware architecture for the final comparison.

### 4.6 Sensor Transformer

The revised repository adds a compact sensor-transformer. It projects each timestep into a learned embedding space, adds learned positional embeddings, applies stacked multi-head self-attention and feed-forward blocks, and classifies the pooled sequence. This creates a foundation-model-inspired architecture while keeping compute requirements realistic for a class project.

### 4.7 Prompt-engineered explanation layer

The Gradio app includes a prompt layer that converts model outputs into a controlled explanation artifact. The prompt includes predicted label, confidence, top probabilities, sensor statistics, and explicit safety instructions such as avoiding medical certainty and mentioning uncertainty. The app also provides a deterministic fallback explanation so the demo works without external API keys.

## 5. Experiments

### 5.1 Baseline architecture comparison

| Model | Test accuracy | Macro F1 | Train time |
|---|---:|---:|---:|
| MLP | 91.38% | 0.9135 | 82s |
| CNN | 91.79% | 0.9189 | 529s |
| CNN-LSTM | 93.25% | 0.9335 | 829s |

The CNN-LSTM is the strongest initial model. The largest expected errors are between SITTING and STANDING because both are static activities with similar low-amplitude signals.

### 5.2 Required final ablations

The repository contains runnable tooling for the following final experiments:

- **Architecture:** MLP vs CNN vs CNN-LSTM vs TCN vs CNN-BiLSTM-Attention vs sensor-transformer.
- **Sensor subset:** all sensors vs body accelerometer vs gyroscope vs total accelerometer.
- **Augmentation:** no augmentation vs jitter/scaling/time masking.
- **Hyperparameters:** learning rate and batch size sweeps.

### 5.3 Evaluation metrics

The primary metrics are accuracy and macro F1. Macro F1 is important because it treats all classes equally. The pipeline also saves per-class F1, confusion matrices, inference latency per window, parameter count, training curves, and model summaries.

### 5.4 Visualization plan

The final submission should include:

- TensorBoard training curves
- MLflow run comparison screenshot
- architecture comparison bar chart
- ablation tables
- normalized confusion matrix
- Gradio inference screenshot
- prompt explanation screenshot
- drift-monitoring screenshot

## 6. MLOps and Application

The project includes a reproducible training pipeline, an evaluation script, saved normalizer artifacts, checkpointing, TensorBoard logging, optional MLflow logging, unit tests, a Dockerfile, GitHub Actions, a DVC-style pipeline definition, drift-reference profile generation, and a Gradio web demo. The Gradio app accepts a CSV or NPY sensor window and returns the predicted activity, confidence, probability table, sensor plot, prompt explanation, and monitoring status.

The implemented system is best described as MLOps maturity level 2: automated training, tracked artifacts, repeatable evaluation, and a deployable demo. A level 3/4 extension would deploy the model through a managed cloud pipeline, monitor production drift, trigger retraining, and auto-promote models only after passing evaluation gates.

## 7. Team Contributions

| Team member | Contribution |
|---|---|
| Vineeth | Model design, training pipeline, baseline notebook, advanced architecture integration, metric interpretation |
| Siddharth | MLOps structure, Docker, CI, MLflow/TensorBoard integration, cloud deployment plan |
| Arshan | Gradio demo, inference workflow, prompt explanation UX, screenshots and demo flow |
| Soham | Ablation studies, sweeps, visualizations, report, slides, error analysis |

## 8. Conclusion

The project shows that deep learning can classify human activity from short smartphone sensor windows and that sequence-aware models improve performance over a flattened baseline. The CNN-LSTM baseline reaches 93.25% test accuracy on unseen subjects, and the revised repo provides a path to compare TCN, attention, and sensor-transformer models. The biggest remaining work before submission is to run final ablations/sweeps, save screenshots, record videos, and publish the public GitHub URL. Future work includes phone-placement robustness, real-time mobile inference, managed cloud deployment, drift monitoring, and automated retraining.
