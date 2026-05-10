# Model Card: DeepHAR-MLOps

## Model family

The repository supports five neural architectures:

1. MLP baseline
2. 1D CNN
3. CNN-LSTM
4. Temporal Convolutional Network (TCN)
5. CNN-BiLSTM-Attention

## Intended use

Classify short smartphone IMU windows into six human activities. The model is intended for educational and research demonstrations of time-series deep learning and MLOps.

## Input

A normalized tensor with shape `batch_size x 128 x channels`, where channels can be all nine UCI HAR channels or selected sensor subsets.

## Output

A probability distribution over six activity labels.

## Metrics

- Accuracy
- Macro F1
- Per-class F1
- Confusion matrix
- Latency during inference

## Known failure modes

- SITTING vs STANDING confusion due to similar low-motion signals.
- Generalization may degrade when phone placement differs from waist-mounted collection.
- The model may be overconfident on out-of-distribution motion signals.

## Ethical and privacy notes

This project avoids GPS, camera, microphone, and personally identifying inputs. Still, activity recognition can reveal behavioral patterns, so production systems should use consent, local processing where possible, and retention limits.
