# Chipless RFID Classification with Deep Learning

Identifying 16 chipless RFID tags from noisy backscattered signals using signal processing, machine learning and deep learning.

This project compares classical baselines with neural networks and combines four architectures into a final ensemble, achieving **97.4% accuracy in stratified 5-fold cross-validation**.

## Overview

Chipless RFID tags encode information through resonant notches in their frequency response, without integrated circuits. In this dataset, five frequency sub-bands between **1.5 and 5 GHz** define a 5-bit code: a notch represents `1`, and its absence represents `0`.

The task is to identify one of **16 tag classes** from measurements affected by noise, tag inclination and distance from the antenna.

The input signals have already been processed using an inverse Fourier transform, time-gating and a **Continuous Wavelet Transform (CWT)**. The notebook works with the resulting CWT representations.

## Dataset

| Dataset | Samples | Purpose |
|---|---:|---|
| Clean references | 80 | Visualisation and reference-based baselines |
| Training | 4,800 | Model training and cross-validation |
| Test | 2,400 | Prediction of unknown labels |

Each measurement contains **1,601 frequency positions × 57 CWT scales**.

The training set is balanced, with **300 samples per class** and 100 samples per class in each distance interval:

- 50–80 cm
- 80–110 cm
- 110–140 cm

Class labels include both the binary code and tag size, for example `tag_01101_1x1`.

### Required data files

- `cwt_references_final.pkl`
- `cwt_training_final.pkl`
- `cwt_testing_final_unlabeled_with_intervals.pkl`

These files must be obtained separately and placed in the folder configured in the notebook.

## Methodology

### Data exploration and preprocessing

The notebook analyses class balance, distance effects, frequency patterns and potential data leakage.

Before training, the frequency axis is downsampled by averaging groups of four points, reducing each input to **400 × 57**. Feature-wise standardisation is fitted using only the training portion of each fold.

### Evaluation

Models are compared using the same **stratified 5-fold cross-validation splits**. Evaluation includes:

- Overall classification accuracy
- Accuracy by distance interval
- Variation across folds
- Out-of-fold predictions and confusion matrices

The test labels are unavailable, so the final test predictions cannot be scored.

### Models and experiments

The study covers:

- Clean-reference matching and class centroids
- Nearest-neighbour classification
- PCA with logistic regression and RBF SVM
- Multilayer perceptrons
- 1D and 2D convolutional neural networks
- Direct class prediction versus 5-bit prediction
- Multi-task learning
- Data augmentation and label smoothing
- Hyperparameter optimisation with Optuna
- EfficientNetV2-B0 with and without ImageNet pretraining
- CNN–Transformer architectures
- Pseudo-labelling and model ensembles

## Results

Rounded accuracy values from the project's 5-fold cross-validation:

| Model | Accuracy |
|---|---:|
| Clean-reference correlation | 61.1% |
| PCA + logistic regression | 89.9% |
| PCA + RBF SVM | 90.3% |
| 1-NN correlation | 91.2% |
| Initial 1D CNN | 96.0% |
| Final 1D CNN | 96.6% |
| EfficientNetV2-B0 with ImageNet pretraining | 96.6% |
| **Final four-architecture ensemble** | **97.4%** |

### Final ensemble

The final model averages class probabilities from:

1. A 1D CNN
2. EfficientNetV2-B0 pretrained on ImageNet
3. EfficientNetV2-B0 trained from scratch
4. A 1D CNN with Transformer layers

For test inference, each architecture contributes predictions from its five fold models, giving **20 trained models** in total.

| Distance interval | Ensemble cross-validation accuracy |
|---|---:|
| 50–80 cm | 99.3% |
| 80–110 cm | 98.2% |
| 110–140 cm | 94.6% |

The ensemble makes **127 errors across 4,800 out-of-fold predictions**.

## Key Findings

- A 1D CNN captures frequency-dependent patterns efficiently and matches the initial MLP with approximately three times fewer parameters.
- Label smoothing provides the clearest consistent improvement to the training recipe.
- Combining architectures improves performance because they make different errors.
- Small `1×1` tags account for approximately 60% of the remaining errors, particularly at longer distances.
- The least-confident 10% of validation predictions contain approximately 96% of the errors.

The notebook also uses **Grad-CAM, t-SNE and confidence analysis** to investigate model behaviour.

## Getting Started

The notebook is configured for **Google Colab with Google Drive**. A GPU is recommended; the recorded experiments used an NVIDIA A100.

### Main dependencies

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- TensorFlow / Keras
- Optuna

### Run the notebook

1. Open `Chipless_RFID_classification_17_11.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Upload the three required `.pkl` datasets to Google Drive.
4. Start at **Section 1 — Setup and data loading** and mount your Drive.
5. Update `path_files` to point to your dataset folder, keeping the trailing `/`.
6. Run the sections sequentially.

The opening cells provide background and dataset illustrations. Section 1 reloads the data independently, so those introductory cells are not required to execute the experiments.

Running the complete notebook trains multiple models across five folds and can take substantial time.

## Generated Outputs

The notebook writes the following files to the configured data folder:

| File | Contents |
|---|---|
| `test_predictions.txt` | 2,400 predicted class names, one per line in the original test order |
| `test_proba_final.npy` | Final ensemble probabilities for the test samples |
| `optuna_best_params_cnn1d.json` | Parameters selected by the Optuna search |

Example prediction:

```text
tag_01101_1x1
```

The notebook retains predictions for ensemble construction rather than exporting a complete set of trained model weights.

## Limitations

- **97.4% is a cross-validation result, not a measured test accuracy.**
- The ensemble was selected using the same cross-validation predictions used to report its performance, which may introduce optimism.
- Random stratified splits may not fully capture generalisation to independent measurement sessions; closely related readings remain a potential concern.
- GPU training is not fully deterministic. Repeated runs can differ by approximately 0.3 percentage points.
- The smallest tags at the longest distances remain the most difficult cases.

## Future Work

- Collect additional measurements of small tags at long distances.
- Evaluate on independent measurement sessions.
- Combine repeated readings or repeat measurements with low confidence.
- Calibrate predicted probabilities using temperature scaling.

## Authors

Beñat Ilzarbe · Iker Elias · Aitor Marcalain
