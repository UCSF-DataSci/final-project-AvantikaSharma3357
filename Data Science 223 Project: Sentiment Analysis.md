# Data Science 223 Project: Sentiment Analysis with BERT (Transformers)

## Overview

In this project, we use **BERT (Bidirectional Encoder Representations from Transformers)** to perform sentiment classification on the [BBC Text dataset](https://www.kaggle.com/datasets/cpichot/bbc-news). The dataset contains 2,225 news articles labeled with one of five categories: `business`, `entertainment`, `politics`, `sport`, and `tech`.

We fine-tuned a pre-trained BERT model (`bert-base-uncased`) for sequence classification using PyTorch and Hugging Face’s `transformers` library.

---

## Dataset

- **Source**: `bbc-text.csv`
- **Structure**: Two columns — `category` (label) and `text` (article content)
- **Label Distribution**:

| Category       | Count |
|----------------|-------|
| Sport          | 511   |
| Business       | 510   |
| Politics       | 417   |
| Tech           | 401   |
| Entertainment  | 386   |

---

## Preprocessing

- **Label Encoding**: Each category is encoded as an integer (e.g., `tech` → 4).
- **Train/Validation Split**: 80/20 stratified split.
- **Tokenization**: Done with `bert-base-uncased` tokenizer using padding and truncation.

---

## Model

- **Architecture**: BERT with a classification head (`BertForSequenceClassification`)
- **Framework**: PyTorch with Hugging Face `transformers`
- **Number of Labels**: 5
- **Optimizer**: `AdamW` with learning rate `1e-5`

---

## Custom Training Loop

Due to issues with the `Trainer` API, we implemented a custom training loop which:
- Performs evaluation before training
- Trains across 3 epochs
- Tracks average loss and validation accuracy per epoch

---

## Results

| Epoch | Average Loss | Validation Accuracy |
|-------|--------------|---------------------|
| 0     | —            | 19.6% (initial)     |
| 1     | 0.4913       | 98.2%               |
| 2     | 0.0631       | 98.7%               |
| 3     | 0.0299       | 99.3%               |

The model showed rapid improvement, achieving strong performance by the second epoch.

---

## Dependencies

- `transformers`
- `datasets`
- `torch`
- `sklearn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `google.colab` (if using Google Drive)


