<a name="readme-top"></a>

# Latent Collaboration in Multi-Agent Systems with KNN Cache Filtering

## 💡 Introduction

This repository is based on the **LatentMAS** framework ([Zou et al., 2025](https://arxiv.org/abs/2511.20639)), a multi-agent reasoning framework that **moves agent collaboration from token space into the model's latent space**.

**Key Features:**

- **Efficient** multi-step reasoning with drastically fewer tokens
- **Training-free** latent-space alignment for stable generation
- **KNN-based KV cache filtering** for memory-efficient agent communication
- **Three selection strategies**: top-k similarity, bottom-k diversity, and random baseline

This implementation extends the original LatentMAS with experimental KNN filtering capabilities for the KV cache, enabling more efficient memory usage during multi-agent collaboration.

## 📊 Supported Datasets

This implementation supports the following datasets:

- **GSM8K**: Grade school math problems
- **GPQA (Diamond)**: Graduate-level science questions
- **MedQA**: Medical question answering

## 🛠️ Getting Started

### ⚙️ Setup Environment Variables

We recommend setting your HF cache directory to avoid repeated downloads:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

Models and datasets will automatically be downloaded into `$HF_HOME`.

### 📦 Install Packages

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas

pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YourRepo/LatentMAS.git
cd LatentMAS
```

### 2. Repository Structure

```
LatentMAS/
│── run.py                 # Main entry for experiments
│── models.py              # Wrapper for HF models + latent realignment
│── methods/
│   ├── baseline.py        # Single-agent baseline
│   ├── text_mas.py        # Token-space multi-agent method
│   └── latent_mas.py      # Latent-space multi-agent (with KNN filtering)
│── prompts.py             # Prompt constructors
│── prompts v2.py          # Updated Prompt constructors for Bottom kNN
│── data.py                # Dataset loaders (GSM8K, GPQA, MedQA)
│── data/                  # Provided data + figures
│── utils.py               # Answer parsing / timeout / helpers
│── example_logs/          # Example logs from LatentMAS
│── requirements.txt
```

## 🧪 Running Experiments

### 🔹 **Baseline (single model)**

```bash
python run.py --method baseline --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 100
```

### 🔹 **TextMAS (text-based multi-agent system)**

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-4B --task gsm8k --prompt sequential --max_samples 100
```

### 🔹 **LatentMAS (latent multi-agent system)**

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task gsm8k --latent_steps 10 --prompt sequential --max_samples 100
```

#### Key Parameters:

- **`--latent_steps`** ∈ [0, 80]
  Number of latent reasoning steps per agent. Typically **10–20** works well.

- **`--latent_space_realign`**
  Enables latent→embedding alignment for better generation stability.

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task gsm8k --latent_steps 10 --latent_space_realign --max_samples 100
```

- **`--prompt`** ∈ {`sequential`, `hierarchical`}
  Prompt structure for agent collaboration.

## 🔬 KNN Cache Filtering (Experimental)

This implementation includes experimental KNN-based filtering of the KV cache to reduce memory usage during agent-to-agent communication.

### Key KNN Parameters:

- **`--knn_filter`**
  Enable KNN filtering of the KV cache

- **`--knn_percentage`** (default: 0.8)
  Percentage of tokens to keep (0.0-1.0). E.g., 0.8 keeps 80% of the cache.

- **`--knn_min_keep`** (default: 5)
  Minimum number of recent tokens to always preserve, regardless of similarity.

- **`--knn_strategy`** ∈ {`top`, `bottom`, `random`} (default: `top`)
  - **`top`**: Keep most similar tokens
  - **`bottom`**: Keep least similar tokens
  - **`random`**: Keep random tokens

### 🧬 KNN Filtering Examples

#### 1. Standard KNN: Keep 80% most similar tokens

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --latent_steps 10 \
  --max_samples 10 \
  --knn_filter \
  --knn_percentage 0.8 \
  --knn_strategy top
```

#### 2. Aggressive filtering: Keep only 50% most similar

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task gpqa \
  --latent_steps 10 \
  --max_samples 10 \
  --knn_filter \
  --knn_percentage 0.5 \
  --knn_strategy top
```

#### 3. Diversity baseline: Keep 80% least similar tokens

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task medqa \
  --latent_steps 10 \
  --max_samples 10 \
  --knn_filter \
  --knn_percentage 0.8 \
  --knn_strategy bottom
```

#### 6. Full experiment with all features

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt hierarchical \
  --latent_steps 20 \
  --max_samples 100 \
  --latent_space_realign \
  --knn_filter \
  --knn_percentage 0.7 \
  --knn_min_keep 5 \
  --knn_strategy top \
  --temperature 0.6 \
  --seed 42
```

## 📚 Citation

This implementation is based on the LatentMAS paper. If you find this work helpful, please cite:

```
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## 🤝 Acknowledgement

This code is based on the LatentMAS framework by [Zou et al., 2025](https://arxiv.org/abs/2511.20639). The KNN cache filtering extension was developed independently for research purposes.
