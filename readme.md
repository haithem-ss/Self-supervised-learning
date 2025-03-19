# Self-Supervised Learning for Image Representation Learning

## Overview
This project explores **Self-Supervised Learning (SSL)** through an **image rotation prediction pretext task**. The goal is to train a model to recognize image orientations, allowing it to learn meaningful representations from unlabeled data. These learned features are later transferred to a **downstream image classification task**.

> **Note:** This project is a proof of concept designed to demonstrate the Self-Supervised Learning technique.

## 🔍 Project Details
Self-supervised learning eliminates the need for labeled datasets by leveraging **pretext tasks** that generate supervisory signals from raw data. In this project, we:
- Train a model to predict **image rotations** (0°, 90°, 180°, 270°) as the pretext task.
- Use **MobileNetV2** as the feature extractor to improve learning efficiency.
- Fine-tune the model for a **binary classification task** (distinguishing between two classes: *Duck* and *Fish*).

## 📂 Dataset
- **Dataset:** [Tiny ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) (subset of ImageNet).
- **Pretext Task:** Image rotation classification.
- **Downstream Task:** Binary classification with selected categories.

## ⚙️ Model Architecture
- **Feature Extractor:** MobileNetV2 (pretrained backbone).
- **Data Augmentation:** Horizontal flips, zoom, contrast adjustments.
- **Classification Head:** Dense layers with ReLU and softmax activation.
- **Loss Function:** Sparse categorical cross-entropy.

## 🚀 Implementation Details
- **Frameworks:** TensorFlow, Keras, NumPy, Matplotlib.
- **Training Strategy:**
  - **Pretext Task:** Train model on rotation prediction.
  - **Fine-Tuning:** Transfer learned features to the classification task.
  - **Optimization:** Adam optimizer, early stopping, learning rate scheduling.

## 📊 Results
- Training from scratch performed poorly due to unstable optimization.
- Using **MobileNetV2** significantly improved performance.
- The fine-tuned model achieved high accuracy in the downstream task.

## 🖥️ Training Environment
- **Platform:** Google Colab.
- **Hardware:** NVIDIA Tesla T4 GPU.
- **Python Version:** 3.9.

## Project Structure
```
Projet DL/
├── models/                 # Saved model checkpoints
├── src/                    # Source code
│   ├── data/               # Data processing scripts
│   ├── models/             # Model architecture definitions
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.7+
- CUDA (for GPU acceleration, optional)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/haithem-ss/Self-supervised-learning
cd "Projet DL"
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## References
[1] YouTube, "Self-Supervised Learning Overview," available at: https://www.youtube.com/watch?v=CG9xbAfq6wI.

[2] Neptune.ai, "Self-Supervised Learning: What It Is and How It Works," available at: https://neptune.ai/blog/self-supervised-learning#:~:text=Self%2Dsupervised%20learning%20is%20a,as%20predictive%20or%20pretext%20learning.

[3] V7 Labs, "The Ultimate Guide to Self-Supervised Learning," available at: https://www.v7labs.com/blog/self-supervised-learning-guide.

[4] Shelf.io, "Self-Supervised Learning Harnesses the Power of Unlabeled Data," available at: https://shelf.io/blog/self-supervised-learning-harnesses-the-power-of-unlabeled-data/.

[5] Kaggle, "Tiny ImageNet," available at: https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet.

[6] Tsang, S., "Review: SimCLR – A Simple Framework for Contrastive Learning of Visual Representations," Medium, available at: https://sh-tsang.medium.com/5de42ba0bc66.

[7] AI Multiple, "Self-Supervised Learning," available at: https://research.aimultiple.com/self-supervised-learning/#what-are-its-limitations.

## More Resources
- Checkout my article about this project [here](https://haithemsaida.tech/blog/ssl).
