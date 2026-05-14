# Architecture Decisions

## Why raw time-series learning?

The project uses raw inertial windows instead of only hand-engineered features so the neural models can learn temporal movement patterns directly. This better matches the course emphasis on model architecture and training loops.

## Why subject-based splitting?

The UCI HAR split evaluates on unseen subjects. This is harder and more realistic than randomly mixing windows from the same people across train and test.

## Why z-score normalization?

Each sensor channel has a different scale. The `ChannelStandardizer` fits mean and standard deviation on the training data only, preventing leakage into validation/test/inference.

## Why these models?

- **MLP:** baseline with no explicit temporal bias.
- **1D CNN:** captures local motion motifs.
- **CNN-LSTM:** combines local feature extraction with sequence memory.
- **TCN:** uses dilated residual convolutions for longer temporal context.
- **CNN-BiLSTM-Attention:** learns bidirectional temporal context and attends to important timesteps.
- **Sensor-transformer:** adds a foundation-model-inspired encoder with token projection, positional embeddings, and self-attention.

## Why sparse categorical cross-entropy?

The labels are mutually exclusive integer classes. Sparse categorical cross-entropy avoids one-hot expansion and matches the softmax output layer.

## Why macro F1 in addition to accuracy?

Accuracy can hide class-specific failures. Macro F1 weights each class equally and is useful for identifying weak classes such as SITTING vs STANDING.

## Why prompt-engineered explanations?

The classifier gives probabilities, but a project demo needs a concise and cautious explanation. The prompt layer provides a reproducible LLM-ready artifact while keeping the core prediction deterministic.
