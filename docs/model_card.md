# Model Card: DeepHAR-MLOps

## Model family

The repository supports six neural architectures:

1. MLP baseline
2. 1D CNN
3. CNN-LSTM
4. Temporal Convolutional Network
5. CNN-BiLSTM-Attention
6. Sensor Transformer

## Intended use

Classify short smartphone IMU windows into six human activities for educational demonstrations of time-series deep learning, MLOps, and prompt-engineered model explanations.

## Not intended for

- medical diagnosis
- emergency fall detection without clinical validation
- surveillance or employee monitoring
- deployment on new phone placements without validation

## Input

A normalized tensor with shape `batch_size x 128 x channels`, where channels can be all nine UCI HAR channels or selected sensor subsets.

## Output

A probability distribution over six activity labels:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## Explanation layer

The app includes a prompt-engineered explanation layer. It does not replace the classifier. It converts prediction context into a controlled prompt and deterministic fallback explanation for a project-demo audience.

## Metrics

- Accuracy
- Macro F1
- Per-class F1
- Confusion matrix
- Inference latency per window
- Parameter count
- Drift score for uploaded windows

## Known failure modes

- SITTING vs STANDING confusion due to similar low-motion signals.
- Generalization may degrade when phone placement differs from waist-mounted collection.
- The model may be overconfident on out-of-distribution motion signals.
- Uploaded files with a different channel order may produce invalid predictions.

## Ethical and privacy notes

This project avoids GPS, camera, microphone, and directly identifying inputs. Still, activity recognition can reveal behavioral patterns, so production systems should use consent, local processing where possible, retention limits, and clear disclosure.
