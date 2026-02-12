This repository contains a collection of advanced Jupyter Notebooks focusing on Natural Language Processing (NLP), Large Language Models (LLMs), and Computer Vision (CV). Each project explores a different facet of modern Deep Learning, from Seq2Seq translation to Transformer-based re-ranking.

##  Project Structure

### 1. Transformer-based Duplicate Detection (`Transformer.ipynb`)
**Task:** Semantic Similarity / Question Answering.
- **Architecture:** Two-stage Search Pipeline (Lexical Retrieval + Transformer Re-ranking).
- **Core Tools:** `transformers`, `datasets`, `deepspeed`, `accelerate`.
- **Logic:** Uses TF-IDF to shortlist candidates and a fine-tuned Transformer to calculate semantic overlap.

### 2. Neural Machine Translation (`LLM.ipynb`)
**Task:** Seq2Seq Russian-to-English Translation.
- **Architecture:** Encoder-Decoder RNN with GRU/LSTM cells.
- **Key Concepts:** BPE Tokenization, Teacher Forcing, and hidden state context vectors.
- **Data:** Parallel RU-EN corpora.

### 3. Transformers Deep Dive (`Huggingface_transformers.ipynb`)
**Task:** NLP Fundamentals & Text Generation.
- **Focus:** Understanding the internal mechanics of Hugging Face pipelines.
- **Implementation:** Manual text generation loop (Logits -> Softmax -> Sampling) and Masked Language Modeling (MLM).

### 4. CNN Fine-Tuning (`Finetuning.ipynb`)
**Task:** Image Classification via Transfer Learning.
- **Architecture:** ResNet-18 (Residual Networks).
- **Dataset:** Pre-trained on ImageNet-1K.
- **Concepts:** Freezing layers, modifying the Fully Connected (FC) head, and learning rate scheduling.

##  Requirements

Install the necessary dependencies to run all notebooks:

```bash
pip install torch torchvision transformers datasets accelerate deepspeed scikit-learn tqdm numpy
```

##  Getting Started
1. Clone the repository.
2. Ensure you have access to a GPU (highly recommended for `Transformer.ipynb` and `LLM.ipynb`).
3. Open the desired notebook in Jupyter Lab or Google Colab.

