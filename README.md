# GBCG: Granular Ball and Counterfactual Guided Profile Injection Attack

This repository contains the official PyTorch implementation for the paper:

> **GBCG: Granular Ball and Counterfactual Guided Profile Injection Attack in Recommender Systems** > *Yunmeng Zhao, Yuran He, Shenbao Yu, Ruihong Huang, Jiayin Li, Jun Shen, Jiayin Lin* > Accepted by **DASFAA 2026**

## 📝 Abstract

GBCG is a generative shilling attack framework designed for **black-box** and **resource-constrained** recommender systems. By integrating **Granular-Ball Computing (GBC)** and **Counterfactual Gain Estimation**, GBCG generates effective and stealthy fake user profiles. Specifically:

- **Granular-Ball Computing**: Constructs a multi-granularity candidate space to capture local semantic structures.
- **Counterfactual Guidance**: Selects high-impact items efficiently without accessing model gradients.
- **WGAN-GP Generator**: Synthesizes realistic user behaviors to evade detection.

## 🔧 Environment Requirements

The code has been tested under the following environment:

- **Python** 3.10
- **PyTorch** 2.4.1
- **numpy** >= 1.26
- **scikit-learn** >= 1.5
- **absl-py** == 2.1.0

## 📂 Data Preparation

We use three public datasets: **MovieLens-100K**, **Yelp**, and **Amazon Automotive**. Please download them from the official links below and place them in the corresponding directories.

### 1. Download Datasets

- **MovieLens 100K**: [Download Link](https://grouplens.org/datasets/movielens/100k/)
- **Yelp**: [Download Link](https://www.kaggle.com/c/yelp-recruiting/data)
- **Amazon Automotive**: [Download Link](https://www.google.com/search?q=https://cseweb.ucsd.edu/~jmcauley/datasets.html%23amazon_reviews) 

### 2. Directory Structure

Extract the files and verify the directory structure is as follows:

```
GBCG/
├── data/
│   └── clean/
│       ├── ml-100k/
│       │   ├── u.data
│       │   └── ...
│       ├── yelp/
│       │   └── ...
│       └── automotive/
│           └── ...
```

## 🚀 Quick Start

### 1. Train the Surrogate Model (LightGCN)

Since GBCG operates in a black-box setting using a surrogate model to extract item embeddings, you must first train a LightGCN model (or use a pre-trained one).

```
# Example command to train LightGCN and save the model
python main.py --model_name LightGCN --dataset ml-100k --save_model True
```

*Ensure the trained model is saved to `modelsaved/LightGCN/`.*

### 2. Run GBCG Attack

After the surrogate model is ready, you can run the GBCG attack. The script will load the saved LightGCN model automatically.

```
python main.py \
  --model_name LightGCN \
  --attackCategory Black \
  --attackModelName GBCG \
  --dataset ml-100k \
  --maliciousUserSize 0.01 \
  --attackTargetChooseWay unpopular
```

### Key Arguments

- `--maliciousUserSize`: Ratio of injected fake users (e.g., `0.01` for 1%).
- `--attackTargetChooseWay`: Strategy to choose target items (`unpopular`, `random`, `popular`).
- `--times`: Number of experimental runs (default: 5).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
