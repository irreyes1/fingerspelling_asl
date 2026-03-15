# Sign Language Fingerspelling Recognition

**Team:** 
- Iñaki Rodriguez 
- Julien Benjamin Cojan 
- Pau Vila 
- Santi Scalzadonna

**Advisor:** 
- Laia Tarres Benet

---

## Table of Contents

- [Sign Language Fingerspelling Recognition](#sign-language-fingerspelling-recognition)
  - [Table of Contents](#table-of-contents)
  - [1. Motivation](#1-motivation)
  - [2. Our Proposal](#2-our-proposal)
    - [2.1 Infrastructure \& MLOps](#21-infrastructure--mlops)
    - [2.2 Dataset](#22-dataset)
    - [2.3 Preprocessing Pipeline](#23-preprocessing-pipeline)
    - [2.4 Neural Architectures](#24-neural-architectures)
    - [2.5 Training Setup](#25-training-setup)
  - [3. Experiments](#3-experiments)
    - [Experiment 1 — Baseline RNN](#experiment-1--baseline-rnn)
    - [Experiment 2 — Architecture Comparison](#experiment-2--architecture-comparison)
    - [Experiment 3 — TCN+BiLSTM Hyperparameter Tuning](#experiment-3--tcnbilstm-hyperparameter-tuning)
  - [4. Next Steps](#4-next-steps)
  - [5. Final Thoughts](#5-final-thoughts)
  - [6. How to Run](#6-how-to-run)
    - [Prerequisites](#prerequisites)
    - [Dataset](#dataset)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Real-Time Webcam Inference](#real-time-webcam-inference)
    - [Quick Inference on a Single Sequence](#quick-inference-on-a-single-sequence)
  - [References](#references)

---

## 1. Motivation

Fingerspelling is a fundamental component of American Sign Language (ASL), used to spell out proper names, technical terms, and words that lack a dedicated sign. Despite its importance, automatic fingerspelling recognition remains an open and challenging problem. Unlike static hand gesture classification — which maps a single image to a letter — fingerspelling recognition requires understanding temporal sequences of hand poses that together form words and phrases. The model must learn not just the shape of each letter, but also the transitions between them, which vary naturally across signers, signing speeds, and recording conditions.

Automatic interpretation of fingerspelling could greatly improve accessibility and inclusion for the deaf and hard-of-hearing community, enabling real-time communication in contexts where human interpreters are unavailable: medical appointments, customer service interactions, emergency situations, and everyday conversations. Existing solutions are either too slow for real-time use, too brittle across signers, or require specialized hardware.

This project explores how deep learning can help bridge this communication gap. Starting from raw hand landmark data extracted by MediaPipe, we build a system that recognizes fingerspelled phrases as text — and we evaluate several neural architectures in terms of both accuracy and practical trainability under constrained compute budgets.

**Goals:**
- Build a complete pipeline from raw hand landmarks to recognized text.
- Establish a reproducible baseline using a simple RNN.
- Progressively improve performance by exploring more expressive architectures.
- Create a foundation that could scale to other sign languages or be extended with richer features.

---

## 2. Our Proposal

### 2.1 Infrastructure & MLOps

The project was developed iteratively across multiple compute environments, driven by resource availability and the need to scale experiments:

- **Language & framework:** Python with PyTorch.
- **Training progression:** We started on Kaggle notebooks for initial prototyping. As experiments grew in complexity, we migrated to Lightning AI for better runtime management. Once Lightning compute credits were exhausted, we moved training to Google Cloud GPU instances for higher throughput and larger dataset runs.
- **MLOps orchestration:** ClearML was configured as the central orchestration layer, enabling experiment tracking, task queuing, and dynamic resource management across environments. This allowed us to run and compare experiments reproducibly regardless of the underlying compute platform.
- **Experiment tracking:** Weights & Biases (W&B) was used to monitor and compare training runs, tracking metrics such as CER, training loss, and average edit distance across all experiments.
- **Hand detection:** MediaPipe was used for real-time hand landmark extraction during inference (webcam demo).

### 2.2 Dataset

We use the **Google ASL Fingerspelling Competition dataset** from Kaggle, which contains sequences of MediaPipe hand landmarks paired with the corresponding text phrases. The dataset covers **94 participants** and, after cleaning, yields approximately **67,000 sequences** of signed phrases.

Each frame in a sequence captures the 3D coordinates of 21 right-hand landmarks (x, y, z), giving 63 raw features per frame. Metadata is provided via `train.csv`, linking each `sequence_id` to its Parquet landmark file and target phrase. The train/val split is done **by participant ID** (80/20) to prevent data leakage — i.e., the same person never appears in both training and validation sets, ensuring the model generalizes to unseen signers.

- **Vocabulary:** 27 characters (a–z + space), plus one CTC blank token → 28 output classes.
- **Data split:** Train: 54,496 sequences · Val: 6,209 · Test: 6,503.

### 2.3 Preprocessing Pipeline

Raw landmark data requires several cleaning and enrichment steps before it can be used for training:

1. **Invalid frame removal:** Frames where all 21 landmarks are NaN (i.e., MediaPipe failed to detect the hand) are dropped entirely. Remaining NaN values in partially detected frames are filled with zero.
2. **Normalization:** Each frame is centered on the wrist landmark (landmark 0) and scaled by the maximum extent across x and y axes. This makes the representation translation- and scale-invariant, so the model is not affected by where the hand appears in the frame or how close the signer is to the camera.
3. **Velocity features:** For each frame, we compute the per-landmark displacement relative to the previous frame (frame-to-frame delta). These velocity features are concatenated to the position features, doubling the input dimensionality: **63 position + 63 velocity = 126 input features**. Velocity captures the dynamic aspect of fingerspelling — how fast the hand is moving — which the position alone does not encode.
4. **Phrase cleaning:** The original dataset contains phrases with digits, punctuation, and URLs (e.g., phone numbers, addresses). These were filtered out to keep the initial vocabulary focused on lowercase letters and spaces only, reducing noise during early training.
5. **Data augmentation (training only):**
   - *Temporal resampling:* Each sequence is randomly resampled to 0.8×–1.2× its original speed, simulating faster and slower signers.
6. **Sequence padding/truncation:** All sequences are padded or truncated to a fixed length of **160 frames** to enable batched training.

### 2.4 Neural Architectures

We evaluated four architectures, progressing from a simple recurrent baseline to more powerful hybrid and attention-based models:

**RNN (Baseline)**
A simple single-direction RNN with a linear input projection layer. Architecture: `Linear(126→128) → RNN(128, 1 layer) → Linear(60) → LogSoftmax`. This model captures basic temporal dependencies and establishes the performance floor. Its simplicity makes it useful for diagnosing data quality issues, gradient instability, and coordinate system problems before committing to more complex models.

**TCN + BiLSTM**
A two-stage hybrid model. First, a stack of Temporal Convolutional Network (TCN) blocks with dilated convolutions extracts local temporal patterns at multiple timescales. The TCN output is then processed by a Bidirectional LSTM, which reads the sequence in both forward and backward directions to provide each timestep with full context from the entire sequence. This combination is well-suited for fingerspelling: the TCN captures the finger shape transitions between adjacent frames, while the BiLSTM handles word-level temporal structure.

**Transformer**
A standard encoder-only Transformer using multi-head self-attention. Self-attention can theoretically model any pairwise dependency between frames, regardless of distance, making it appealing for long fingerspelled phrases. However, Transformers are data-hungry and typically need significantly more training examples and epochs to learn useful representations from scratch.

**Conformer**
A hybrid of convolution and self-attention, originally proposed for speech recognition (ASR). It interleaves local convolutional blocks with global self-attention blocks to capture both fine-grained local patterns and long-range context. Like the Transformer, it tends to require large datasets to show its full potential.

### 2.5 Training Setup

All models were trained with the following common setup:

- **Loss function:** CTC (Connectionist Temporal Classification). CTC is the standard choice for sequence-to-sequence tasks where the alignment between input frames and output characters is unknown. It learns to assign character probabilities across frames without requiring frame-level labels.
- **Decoding:** Greedy CTC decoding at inference time — the most likely character per frame is selected, then consecutive duplicates and blank tokens are collapsed.
- **Optimizer:** Adam with an initial learning rate of 5e-4 (adjusted per experiment).
- **Regularization:** Gradient clipping and early stopping based on validation CER.
- **Primary metric:** Character Error Rate (CER) = (Insertions + Deletions + Substitutions) / Total characters. Lower is better; a CER of 0 means perfect prediction.

---

## 3. Experiments

The experiments were conducted in three stages: establishing a baseline, comparing architectures, and tuning the best architecture.

---

### Experiment 1 — Baseline RNN

**Hypothesis:** A simple unidirectional RNN trained on raw landmark positions is sufficient to learn some structure from the data and provide a meaningful performance baseline. We expect it to underfit due to limited model capacity, but it should confirm that the data pipeline and CTC training setup are working correctly.

**Setup:**
- Architecture: Linear(126→128) → RNN(128, 1 layer) → Linear(60) → LogSoftmax
- Dataset: 54,496 training sequences (full cleaned dataset)
- Epochs: 20, Batch size: 16, LR: 5e-4, Hidden dim: 128

**Results:**
- Training loss: 2.1
- Validation CER: 0.70

**Conclusions:**
The RNN learns to reduce CER steadily throughout training, confirming that the pipeline is functional and the data contains learnable structure. However, CER 0.70 means the model is still making errors on roughly 7 out of every 10 characters — it can partially recognize common or short sequences but fails on longer or less frequent phrases. The unidirectional nature of the RNN limits its ability to use future context when decoding each frame. This motivated exploring architectures with bidirectional processing and multi-scale feature extraction.

---

### Experiment 2 — Architecture Comparison

**Hypothesis:** More expressive architectures — particularly those that combine local feature extraction with bidirectional context — will outperform the simple RNN. Transformer-based models may struggle under our data and compute constraints.

**Setup:**
All four architectures (RNN, TCN+BiLSTM, Transformer, Conformer) were trained under the same conditions to enable a fair comparison:
- Dataset: same 3k-sequence subset for initial comparison
- Epochs: 20, Batch size: 16, LR: 5e-4

**Results:**

| Architecture    | Train Loss | Val CER | Notes                              |
|-----------------|------------|---------|------------------------------------|
| RNN             | 2.1        | 0.70    | Stable training, limited capacity  |
| TCN + BiLSTM    | **1.6**    | **0.60**| Best performance, stable convergence|
| Conformer       | 3.1        | 0.90    | Early stopping, emits many blanks  |
| Transformer     | 3.2        | 1.00    | Early stopping, fails to converge  |

**Conclusions:**
TCN+BiLSTM is the clear winner in this constrained setting. The combination of dilated convolutions for local pattern extraction and bidirectional LSTM for sequence-level context provides the best balance of capacity and data efficiency. The RNN remains a functional baseline. Both the Transformer and Conformer trigger early stopping — they fail to reduce CER meaningfully within 20 epochs on the small subset. This is expected: attention-based models are known to require more data and longer training to stabilize. Their underperformance here is not a reflection of their true capability, but rather of the resource constraints of this study. TCN+BiLSTM was selected as the architecture to optimize in Experiment 3.

---

### Experiment 3 — TCN+BiLSTM Hyperparameter Tuning

**Hypothesis:** The TCN+BiLSTM architecture has not yet reached its performance ceiling. Increasing training data, model capacity, and training duration should yield substantial further improvements in CER.

**Setup:**
Three configurations were compared progressively:

| Parameter     | v1 (Baseline) | v2     | v3 (Best) |
|---------------|--------------|--------|-----------|
| Epochs        | 20           | 40     | **50**    |
| Train size    | 3k           | 50k    | **50k**   |
| Batch size    | 16           | 16     | **32**    |
| Learning rate | 5e-4         | 5e-4   | **1e-3**  |
| Hidden dim    | 128          | 128    | **256**   |

Key changes from v1 to v3:
- Training data increased from 3k to 50k sequences (~16× more).
- Hidden dimension doubled from 128 to 256, increasing model capacity.
- Batch size increased from 16 to 32 for more stable gradient estimates at larger scale.
- Learning rate raised to 1e-3 to accelerate convergence over more epochs.

**Results:**

| Metric              | v1    | v2    | v3 (Best) |
|---------------------|-------|-------|-----------|
| Train Loss          | 1.57  | 1.22  | **0.71**  |
| Val CER             | 0.63  | 0.52  | **0.38**  |
| Avg Edit Distance   | 9.87  | 8.44  | **4.95**  |

**Conclusions:**
The most impactful single change was **scaling the training data from 3k to 50k sequences**. This alone (v2) reduced CER by 0.11 points. Further increasing model capacity and tuning learning rate and batch size (v3) brought CER down to 0.38 — a 40% relative improvement over the initial v1 configuration. The average edit distance dropped from ~10 characters off per prediction to ~5, which starts to approach practically useful quality for shorter phrases. The learning curves show that the model had not converged by epoch 20 in earlier runs, confirming that longer training was warranted.

**Qualitative example (v3):**

| Ground Truth        | Prediction        | CER  |
|---------------------|-------------------|------|
| 5449 clayton creek  | 49 claytn crek    | 22%  |
| leona owens         | leona owens       | 0%   |
| tampa fl            | tampa fl          | 0%   |
| alexis dorsey       | jan               | 100% |

The model handles short, common phrases well but still struggles with longer or less frequent sequences. The word "clayton" predicted as "claytоn" (one deletion) illustrates CER computation: 1 edit / 7 characters = 14.3%.

---

## 4. Next Steps

The current system is a working proof of concept. Several directions offer clear paths toward production-quality performance:

**Dataset & Feature Enrichment**
- **Pairwise fingertip distances:** Computing the Euclidean distance between each pair of the 5 fingertips per frame would add 10 new features that explicitly encode the spatial relationship between fingers. This could help the model distinguish between letters that differ only in relative finger position (e.g., 'a' vs 'e').
- **Vocabulary expansion:** The current model only handles lowercase letters and spaces. Expanding to include digits and punctuation would greatly increase practical applicability, at the cost of a larger output space and more training data requirements.
- **Larger datasets:** The Transformer and Conformer architectures were clearly undertrained in our experiments. Training them on a significantly larger corpus (e.g., combining multiple ASL datasets) could unlock their potential and surpass the TCN+BiLSTM.

**Architecture Improvements**
- **Graph Neural Networks (GNNs):** The 21 hand landmarks have a natural graph structure (finger joints connected along each finger, with the palm as a hub). A GNN layer before the BiLSTM could explicitly model these spatial relationships, learning which landmark connections are most informative for each letter. This is a promising direction for improving the spatial feature representation.
- **Transformer with pretraining:** Rather than training a Transformer from scratch on limited data, fine-tuning a model pretrained on a large motion or gesture dataset could make attention-based architectures viable in this setting.
- **Beam search decoding:** Replacing greedy CTC decoding with beam search — optionally combined with a character-level language model — could reduce word-level errors, especially for ambiguous or partially corrupted predictions.

**Real-Time Inference**
- The current webcam demo (`realtime_webcam_infer.py`) provides a working real-time pipeline using MediaPipe for landmark extraction and the trained TCN+BiLSTM checkpoint for recognition. Improving latency and robustness to different lighting conditions and hand orientations would be key for a deployable system.

---

## 5. Final Thoughts

This project delivered a complete ASL fingerspelling recognition pipeline — from raw hand landmark sequences to predicted text — built and iterated across a realistic MLOps stack (ClearML, W&B, Google Cloud, Kaggle, Lightning AI).

Starting from a simple RNN that scored CER 0.70, we systematically explored four architectures, identified TCN+BiLSTM as the most effective under constrained resources, and reduced validation CER to **0.38** through dataset scaling and hyperparameter tuning. Beyond the metrics, the project gave the team hands-on experience with the full lifecycle of a deep learning system: dataset curation and cleaning, feature engineering for structured sequential data, CTC-based sequence modeling, multi-platform training infrastructure, and the practical tradeoffs between model capacity, data size, and compute budget.

The gap between our current results and state-of-the-art fingerspelling systems is primarily a function of data and compute — not architecture design. The foundation is solid. With richer input features, more training data, and architectures like GNNs or large pretrained Transformers, we believe there is significant room to push CER well below 0.20, approaching the level needed for real-world deployment.

---

## 6. How to Run

### Prerequisites

```bash
git clone <repo_url>
cd fingerspelling_asl
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `numpy`, `pandas`, `pyarrow`, `mediapipe`, `wandb`, `tqdm`, `tensorboard`, `opencv-python`, `streamlit`, `plotly`

### Dataset

Download the Google ASL Fingerspelling dataset from Kaggle and place the files as follows:

```
fingerspelling_asl/
  data/
    train.csv
    train_landmarks/
      <file_id>.parquet
      ...
```

### Training

```bash
cd fingerspelling_asl

# Train the TCN+BiLSTM model (best configuration)
python -m src.train \
  --model lstm \
  --train_csv data/train.csv \
  --data_dir data/asl-fingerspelling \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --hidden_dim 256 \
  --train_size 50000 
  --val_size 50000


# Train the baseline RNN
python -m src.train \
  --model rnn \
  --train_csv data/train.csv \
  --data_dir data/asl-fingerspelling \
  --epochs 20 \
  --batch_size 16 \
  --lr 5e-4
```

Optionally pass `--wandb_project <project_name>` and `--wandb_tags <tag1,tag2>` to log runs to Weights & Biases.

### Evaluation

```bash
python -m src.evaluate \
       --ckpt artifacts/models/run_best.pt \
       --data_dir data/asl-fingerspelling
```

### Real-Time Webcam Inference

```bash
python -m src.realtime_webcam_infer \
  --checkpoint checkpoints/<run_name>/best_model.pt
```

Requires a webcam. MediaPipe will extract hand landmarks in real time and the model will output the predicted fingerspelled text.

### Quick Inference on a Single Sequence

```bash
python -m src.quick_infer \
  --checkpoint checkpoints/<run_name>/best_model.pt \
  --parquet data/train_landmarks/<file_id>.parquet
```

---

## References

- *W&B Workspace report: https://wandb.ai/inaki-rodriguez-reyes-upc-universidad-peruana-de-ciencia/fingerspelling_asl*

- *ClearML: https://app.clear.ml/projects/f7947bf18c6d48039162f95680b94cab/tasks/9a635b0548e941dbab846fd54d52826d/hyper-params/hyper-param/Args*