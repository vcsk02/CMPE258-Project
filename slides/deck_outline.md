# Slide Deck Outline

## Slide 1: Title

DeepHAR-MLOps: Smartphone Human Activity Recognition  
Vineeth, Siddharth, Arshan, Soham

## Slide 2: Motivation

- Smartphones already collect motion signals.
- HAR supports fitness, fall detection, elderly care, and health monitoring.
- No GPS, camera, or manual input is required.

## Slide 3: Problem Statement

Input: 2.56 second sensor window, 128 timesteps x 9 channels.  
Output: one of six activities.  
Goal: generalize to unseen subjects.

## Slide 4: Dataset

- UCI HAR dataset
- 30 volunteers
- Accelerometer and gyroscope
- Subject-based split

## Slide 5: Pipeline

Download data -> normalize -> train -> evaluate -> save artifacts -> Gradio inference.

## Slide 6: Models

- MLP baseline
- 1D CNN
- CNN-LSTM
- TCN
- CNN-BiLSTM-Attention

## Slide 7: Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| MLP | 91.38% | 0.9135 |
| CNN | 91.79% | 0.9189 |
| CNN-LSTM | 93.25% | 0.9335 |

## Slide 8: Error Analysis

Most common confusion: SITTING vs STANDING.  
Walking classes are easier due to periodic motion.

## Slide 9: MLOps and Demo

- Reusable scripts
- Saved preprocessing artifact
- TensorBoard/MLflow hooks
- Dockerfile
- Gradio app
- CI tests

## Slide 10: Team Contributions and Future Work

- Vineeth: models and training
- Siddharth: MLOps
- Arshan: demo
- Soham: experiments and documentation

Future work: real-time mobile inference, drift detection, phone placement robustness.
