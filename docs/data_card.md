# Data Card: UCI Human Activity Recognition Dataset

## Dataset

UCI Human Activity Recognition Using Smartphones.

## Collection setup

The dataset contains smartphone inertial signals collected from volunteers wearing a Samsung Galaxy S II on the waist. The raw signals were sampled at 50 Hz and segmented into 2.56 second windows of 128 timesteps with 50 percent overlap.

## Inputs

Each example is a tensor of shape `128 x 9`:

1. body_acc_x
2. body_acc_y
3. body_acc_z
4. body_gyro_x
5. body_gyro_y
6. body_gyro_z
7. total_acc_x
8. total_acc_y
9. total_acc_z

## Labels

Six activity classes:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## Split

The original dataset uses a subject-based train/test split. This matters because the test set contains people not seen during training, making the evaluation closer to real deployment.

## Preprocessing

The training script fits a per-channel z-score normalizer on the training split only. The same mean and standard deviation are applied to validation, test, and inference windows.

## Limitations

- The dataset uses one phone model and waist placement.
- Real-world users may carry phones in pockets, bags, hands, or different body positions.
- Some static classes, especially SITTING and STANDING, are difficult to separate because the signals are similar.
