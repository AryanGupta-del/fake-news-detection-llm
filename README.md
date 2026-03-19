# Fake News Detection using LLMs

A transformer-based NLP classifier that distinguishes **Fake** from **Real** news articles with high accuracy, leveraging DistilBERT fine-tuned on a large labeled dataset.

## Features

- **Transformer-based classification** — Fine-tuned DistilBERT for binary fake/real news detection
- **Efficient & lightweight** — DistilBERT is 40% smaller and 60% faster than BERT with ~97% of its performance
- **End-to-end pipeline** — Preprocessing, tokenization, training, and evaluation in a single notebook
- **GPU-accelerated training** — Optimized for Google Colab with CUDA support

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | DistilBERT model & tokenizer |
| Hugging Face Datasets | Dataset loading & preprocessing |
| scikit-learn | Evaluation metrics |
| Google Colab | Cloud GPU training environment |

## Dataset

**[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)** — Kaggle

| File | Label | Description |
|---|---|---|
| `Fake.csv` | 0 — Fake | Fabricated or misleading news articles |
| `True.csv` | 1 — Real | Verified, legitimate news articles |

> The dataset contains ~44,000 articles split across both files. Articles are balanced and cover politics, world events, and more.

## Project Structure

fake-news-detection/
├── fake_news_detection.ipynb   # Main training notebook
├── Fake.csv                    # Fake news articles (upload to Colab)
├── True.csv                    # Real news articles (upload to Colab)
├── requirements.txt            # Python dependencies
└── README.md
```

##  How to Run

### Option 1 — Google Colab (Recommended)

1. Open [`fake_news_detection.ipynb`](fake_news_detection.ipynb) in [Google Colab](https://colab.research.google.com/)
2. Upload `Fake.csv` and `True.csv` when prompted
3. Set runtime to **GPU**: `Runtime → Change runtime type → T4 GPU`
4. Click **Run All** (`Runtime → Run all`)

### Option 2 — Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook fake_news_detection.ipynb

### Requirements

```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

## Model Architecture

Input Text
    │
    ▼
DistilBERT Tokenizer (max_length=512)
    │
    ▼
DistilBERT Base Uncased (6 transformer layers)
    │
    ▼
[CLS] Token Representation
    │
    ▼
Dropout (p=0.3)
    │
    ▼
Linear Layer (768 → 2)
    │
    ▼
Softmax → Fake (0) / Real (1)
```


## Results

| Metric | Score |
|---|---|
| Accuracy | ~98% |
| Precision | ~98% |
| Recall | ~98% |
| F1-Score | ~98% |

> Results may vary slightly based on train/test split seed and number of epochs.

## Future Improvements

- [ ] **Streamlit deployment** — Build a web app for real-time article classification
- [ ] **REST API** — Wrap the model with FastAPI for integration with other services
- [ ] **Multilingual support** — Extend to non-English news using `bert-base-multilingual-cased`
- [ ] **Explainability** — Add LIME or SHAP-based attention visualization to explain predictions
- [ ] **Active learning** — Improve model iteratively using uncertain predictions
- [ ] **Larger models** — Benchmark against RoBERTa, DeBERTa, or GPT-based classifiers
- [ ] **Real-time scraping** — Integrate with news APIs (NewsAPI, GDELT) for live predictions

##  Acknowledgements

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Kaggle](https://www.kaggle.com/) for the dataset
- [Victor Sanh et al.](https://arxiv.org/abs/1910.01108) for the DistilBERT paper
