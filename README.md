# AgriAssist: Multimodal Crop Disease Diagnosis and Localized Advisory System

## Overview
AgriAssist is a research-oriented multimodal AI system for crop disease diagnosis and grounded agricultural advisory retrieval.

The project combines:
- CNN-based crop disease classification
- Retrieval-Augmented Generation (RAG)
- Grad-CAM explainability
- Streamlit deployment
- Lightweight DDPM synthetic image generation
- ResNet vs Vision Transformer comparison
- Configurable dataset ingestion pipelines

---

# Core Features

## 1. Crop Disease Classification
- PyTorch CNN classifier
- Image upload prediction
- Confidence scoring
- Evaluation metrics

## 2. Working RAG Pipeline
- Agricultural knowledge base ingestion
- TF-IDF vectorization
- Cosine similarity retrieval
- Top-k advisory retrieval
- Grounded advisory generation

## 3. Explainable AI
- Grad-CAM heatmap generation
- Visual disease region explanation

## 4. Synthetic Data Generation
- Lightweight DDPM implementation
- Noise prediction training
- Synthetic crop image generation

## 5. Backbone Comparison
- ResNet18 training
- Vision Transformer (ViT-B16) training
- Validation accuracy comparison

---

# Project Structure

```text
src/
├── 01_create_sample_dataset.py
├── 02_data_pipeline.py
├── 03_train_initial_model.py
├── 04_rag_advisory_demo.py
├── 05_synthetic_data_demo.py
├── 07_model_evaluation.py
├── 08_gradcam_explainability.py
├── 09_full_rag_pipeline.py
├── 10_build_large_kb.py
└── 11_model_backbone_comparison.py
```

---

# Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Run Full Pipeline

## Create Dataset

```bash
python3 src/01_create_sample_dataset.py
```

## Data Pipeline

```bash
python3 src/02_data_pipeline.py
```

## CNN Training

```bash
python3 src/03_train_initial_model.py
```

## Model Evaluation

```bash
python3 src/07_model_evaluation.py
```

## Grad-CAM Explainability

```bash
python3 src/08_gradcam_explainability.py
```

## Full RAG Pipeline

```bash
python3 src/09_full_rag_pipeline.py
```

## Build Large Knowledge Base

```bash
python3 src/10_build_large_kb.py
```

## ResNet vs ViT Comparison

```bash
python3 src/11_model_backbone_comparison.py
```

## DDPM Synthetic Generation

```bash
python3 src/05_synthetic_data_demo.py
```

## Streamlit Deployment

```bash
streamlit run streamlit_app.py
```

---

# Research Direction

Future upgrades may include:
- FAISS / ChromaDB vector search
- Sentence-transformer embeddings
- Full semantic RAG retrieval
- Mobile deployment
- Multilingual farmer voice assistant
- Production-scale diffusion models

---

# Project Type

This project is intended as a research-oriented academic AI prototype demonstrating multimodal agricultural intelligence pipelines.
