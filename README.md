# Manual Embeddings

Word embeddings implemented from scratch, then applied to a downstream text classification task and compared against pretrained embeddings.

## Contents

| Notebook | Description |
|---|---|
| `word2vec.ipynb` | Implements **Word2Vec (Skip-Gram with negative sampling)** from scratch on the `text8` corpus: builds a vocabulary, applies subsampling of frequent words, generates center/context training pairs, samples negatives (using the `count^0.75` unigram distribution), and trains a custom `SkipGram` model. |
| `imdb-classification-custom.ipynb` | Trains a text classifier on the IMDB sentiment dataset using the custom embeddings learned above. |
| `imdb-classification-GloVe.ipynb` | Trains the same classifier using pretrained **GloVe** embeddings, for comparison against the custom-trained embeddings. |

## Dataset

- `text8` corpus for training the Word2Vec embeddings
- **IMDB** movie review dataset (via Hugging Face `datasets`) for the downstream classification task

## Technologies

- Python
- PyTorch
- Hugging Face `datasets`

## Getting Started

```bash
git clone https://github.com/shehryar11w/manualEmbedding.git
cd manualEmbedding
pip install torch datasets
```

Run `word2vec.ipynb` first to train the custom embeddings, then either classification notebook.

## Learning Outcomes

- How Skip-Gram with negative sampling works internally (subsampling, context windows, negative sampling distribution)
- How embedding quality affects downstream task performance, by comparing custom-trained vs. pretrained (GloVe) embeddings

## License

Educational / personal use.
