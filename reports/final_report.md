# DeepHAR-MLOps: Sequence-Aware Deep Learning for Smartphone Human Activity Recognition

**Authors:** Vineeth, Siddharth, Arshan, Soham

## Abstract

Human activity recognition from smartphone inertial sensors enables fitness tracking, fall detection, elderly care, and context-aware mobile applications without requiring GPS, cameras, or manual input. This project builds an end-to-end deep learning pipeline for classifying six human activities from 2.56 second windows of raw accelerometer and gyroscope signals. We compare a flattened MLP baseline with sequence-aware architectures including 1D CNN, CNN-LSTM, Temporal Convolutional Network, and CNN-BiLSTM-Attention. The original notebook baseline achieves 91.38 percent accuracy with MLP, 91.79 percent with CNN, and 93.25 percent with CNN-LSTM on the subject-based UCI HAR test split. These results support the central hypothesis that temporal inductive bias improves cross-subject activity recognition. The final system includes reproducible training scripts, saved preprocessing artifacts, evaluation reports, ablation support, MLflow/TensorBoard hooks, and a Gradio inference demo.

## 1. Introduction

Smartphones are equipped with motion sensors that continuously measure acceleration and angular velocity. These signals can be used to infer whether a user is walking, sitting, standing, or lying down. Unlike GPS or camera-based approaches, inertial sensing is lower power and less privacy invasive.

The challenge is that sensor data is sequential. A walking signal is not defined only by individual numeric values; it is defined by periodic changes over time. Therefore, models that understand local and temporal patterns should perform better than models that treat the input mostly as a flattened vector.

This project studies that question using the UCI HAR dataset and a subject-based evaluation protocol. We evaluate models on people not seen during training, which is closer to real-world deployment.

## 2. Related Work

Human activity recognition has traditionally used hand-engineered time and frequency features with classical machine learning methods. Deep learning methods reduce the need for manual feature design by learning directly from raw signals. CNNs learn local movement motifs, recurrent models learn longer temporal dependencies, TCNs capture sequence structure using dilated causal convolutions, and attention layers can focus on important timesteps.

Our approach is similar to prior deep HAR systems in that it learns from raw IMU windows. It differs from a simple notebook-only experiment by packaging the work as a reusable training and inference pipeline with MLOps artifacts.

## 3. Data

The project uses the UCI Human Activity Recognition Using Smartphones dataset. Each sample is a 128 timestep window sampled at 50 Hz, corresponding to 2.56 seconds. There are nine sensor channels: body acceleration x/y/z, body gyroscope x/y/z, and total acceleration x/y/z.

The six labels are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING. The original split separates train and test subjects, so the test set contains people not used during training.

Preprocessing uses per-channel z-score normalization. The mean and standard deviation are fitted on the training data only and reused for validation, test, and inference.

## 4. Methods

### 4.1 MLP baseline

The MLP flattens the sensor window and passes it through dense layers. It is useful as a baseline because it can learn nonlinear patterns but has no explicit temporal inductive bias.

### 4.2 1D CNN

The CNN applies temporal convolutions over the sensor channels. This lets the model learn local motion motifs such as periodic acceleration patterns during walking.

### 4.3 CNN-LSTM

The CNN-LSTM first extracts local temporal features with convolution layers, then feeds the resulting sequence into an LSTM. This combines local pattern recognition with sequence modeling.

### 4.4 TCN

The Temporal Convolutional Network uses residual dilated convolution blocks. It captures longer temporal context while preserving efficient parallel training.

### 4.5 CNN-BiLSTM-Attention

The attention model uses convolution, bidirectional LSTM, and multi-head self-attention. This provides a stronger sequence-aware architecture for the final comparison.

## 5. Experiments

### 5.1 Baseline architecture comparison

| Model | Test accuracy | Macro F1 | Train time |
|---|---:|---:|---:|
| MLP | 91.38% | 0.9135 | 82s |
| CNN | 91.79% | 0.9189 | 529s |
| CNN-LSTM | 93.25% | 0.9335 | 829s |

The CNN-LSTM is the strongest initial model. The largest errors are between SITTING and STANDING, which is expected because both are static activities with similar low-amplitude signals.

### 5.2 Ablation studies

Planned ablations include:

- Sensor subset: all sensors vs body accelerometer vs gyroscope vs total acceleration.
- Architecture: MLP vs CNN vs CNN-LSTM vs TCN vs attention.
- Augmentation: no augmentation vs jitter/scaling/time masking.
- Hyperparameters: learning rate, dropout, batch size, hidden size.

### 5.3 Evaluation metrics

The primary metrics are accuracy and macro F1. Macro F1 is important because it treats all classes equally. We also inspect per-class F1 and confusion matrices to understand failure modes.

## 6. MLOps and Application

The project includes a reproducible training pipeline, an evaluation script, saved normalizer artifacts, checkpointing, TensorBoard logging, optional MLflow logging, unit tests, a Dockerfile, and a Gradio web demo. The Gradio app accepts a CSV or NPY sensor window and returns the predicted activity, confidence, probability table, and sensor plot.

## 7. Team Contributions

| Team member | Contribution |
|---|---|
| Vineeth | Model design, training pipeline, baseline notebook, advanced architecture integration |
| Siddharth | MLOps structure, Docker, CI, MLflow/TensorBoard integration |
| Arshan | Gradio demo, inference workflow, UX and demo screenshots |
| Soham | Ablation studies, visualizations, report, slides, error analysis |

## 8. Conclusion

The project shows that deep learning can classify human activity from short smartphone sensor windows and that sequence-aware models improve performance over a flattened baseline. The CNN-LSTM baseline reaches 93.25 percent test accuracy on unseen subjects, and the repo provides a path to stronger TCN and attention-based experiments. Future work includes phone-placement robustness, real-time streaming inference, drift monitoring, and deployment to a cloud MLOps platform.
