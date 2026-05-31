# AgriAssist

## Website Name

**AgriAssist: AI Agricultural Advisory Website**

# AgriAssist

## Local Website Links

Run the website locally using FastAPI:

```text
http://127.0.0.1:8000
```

## GitHub Repository

```text
https://github.com/SohanVarma/AgriAssist-Completed
```

---

## Project Overview

AgriAssist is a multimodal AI agricultural advisory system built as a deployable website. The project helps users diagnose crop-related problems using either text queries or crop image uploads.

The system currently combines:

* Crop disease image classification
* Retrieval-Augmented Generation
* OpenAI-powered advisory generation
* Grad-CAM explainability
* Backbone comparison report
* FastAPI-based website interface

AgriAssist is designed as an academic AI prototype for crop disease diagnosis and evidence-grounded agricultural decision support.

---

## Current Website Features

### 1. Crop Disease Classification

The website supports crop image upload for disease/class prediction.

Current functionality:

* PyTorch CNN classifier
* Image upload prediction
* Predicted disease/class display
* Confidence score display
* Optional advisory generation after prediction

Relevant file:

```text
src/13_predict_uploaded_image.py
```

The trained model is expected at one of these paths:

```text
results/initial_model.pt
models/cnn_crop_disease_model.pt
```

---

### 2. RAG-Based Agricultural Advisory

The website supports text-based agricultural questions.

Current functionality:

* Agricultural knowledge base ingestion
* TF-IDF vectorization
* Cosine similarity retrieval
* Keyword-overlap scoring
* Top-k retrieved document display
* OpenAI-generated advisory report
* Safety note generation

Relevant files:

```text
src/09_full_rag_pipeline.py
src/10_build_large_kb.py
src/14_build_document_kb.py
```

Example queries:

```text
tomato leaf curl whitefly management
brinjal equipment problem
wheat disease treatment
```

---

### 3. OpenAI-Powered Detailed Advisory Report

After retrieving the most relevant documents, AgriAssist sends the retrieved evidence to the OpenAI API and generates a detailed agricultural advisory report.

The generated report includes:

* Query understanding
* Retrieved evidence summary
* Crop and problem diagnosis
* Severity and urgency assessment
* Detailed action plan
* Product or treatment guidance
* Region and season considerations
* Practical farmer checklist
* Evidence limitations
* Final advisory
* Safety note

The OpenAI API key is loaded from a local `.env` file.

---

### 4. Explainable AI with Grad-CAM

The website includes Grad-CAM explainability for uploaded crop images.

Current functionality:

* Upload crop image
* Run CNN prediction
* Generate Grad-CAM heatmap
* Show original image
* Show heatmap visualization
* Explain which image regions influenced the model prediction

Relevant file:

```text
src/15_gradcam_uploaded_image.py
```

---

### 5. Backbone Comparison

The website includes a backbone comparison module.

Current functionality:

* ResNet18 vs Vision Transformer comparison report
* Architecture strengths and limitations
* Deployment suitability comparison
* Explainability compatibility discussion

Relevant file:

```text
src/16_backbone_comparison_web.py
```

---

## Website Stack

The website is built using:

* FastAPI
* Jinja2 templates
* HTML
* CSS
* Uvicorn

Relevant website files:

```text
web_app.py
web/templates/index.html
web/static/style.css
```

---

## Project Structure

```text
AgriAssist-Completed/
├── web_app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── web/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── outputs/
│
├── src/
│   ├── 01_create_sample_dataset.py
│   ├── 02_data_pipeline.py
│   ├── 03_train_initial_model.py
│   ├── 04_rag_advisory_demo.py
│   ├── 05_synthetic_data_demo.py
│   ├── 06_download_real_dataset.py
│   ├── 06_real_dataset_loader.py
│   ├── 07_dynamic_dataset_loader.py
│   ├── 07_model_evaluation.py
│   ├── 08_gradcam_explainability.py
│   ├── 09_full_rag_pipeline.py
│   ├── 10_build_large_kb.py
│   ├── 11_model_backbone_comparison.py
│   ├── 12_unet_ddpm_generation.py
│   ├── 13_predict_uploaded_image.py
│   ├── 14_build_document_kb.py
│   ├── 15_gradcam_uploaded_image.py
│   └── 16_backbone_comparison_web.py
│
├── data/
│   ├── raw_farmer_dataset.csv
│   ├── advisory_knowledge_base_large.json
│   └── expert_advisory_kb.json
│
├── results/
├── models/
└── uploads/
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SohanVarma/AgriAssist-Completed.git
cd AgriAssist-Completed
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
nano .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not upload `.env` to GitHub.

---

## Build the Knowledge Base

Run:

```bash
python3 src/10_build_large_kb.py
```

If PDF documents are added to the project, build document chunks first:

```bash
python3 src/14_build_document_kb.py
python3 src/10_build_large_kb.py
```

The main knowledge base is stored at:

```text
data/advisory_knowledge_base_large.json
```

---

## Train the CNN Model

Run the data pipeline:

```bash
python3 src/02_data_pipeline.py
```

Train the model:

```bash
python3 src/03_train_initial_model.py
```

The model is saved to:

```text
results/initial_model.pt
models/cnn_crop_disease_model.pt
```

---

## Run the Website Locally

Start the FastAPI server:

```bash
uvicorn web_app:app --reload
```

Open the website:

```text
http://127.0.0.1:8000
```

The text advisory form submits to:

```text
http://127.0.0.1:8000/advisory
```

---

## Website Functionalities

### Text Advisory

Use the text form to enter a crop or farming problem.

Example:

```text
tomato leaf curl whitefly management
```

The website retrieves relevant agricultural documents and generates a detailed advisory report.

### Image Prediction

Use the crop image upload form.

The website predicts the disease/class and displays a confidence score.

### Grad-CAM Explainability

Use the Grad-CAM upload form.

The website generates a heatmap showing image regions that influenced the CNN prediction.

### Backbone Comparison

Use the backbone comparison button.

The website displays a ResNet18 vs Vision Transformer comparison report.

---

## Deployment

The project can be deployed as a FastAPI web service.

For Render deployment:

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn web_app:app --host 0.0.0.0 --port $PORT
```


## Safety Disclaimer

AgriAssist is an AI-assisted agricultural decision-support prototype. It should not replace certified agricultural officers, agronomists, or local extension experts.

The system does not provide final pesticide dosage, chemical prescriptions, or guaranteed disease diagnosis. Farmers should verify all pesticide, fungicide, fertilizer, irrigation, equipment, and treatment recommendations with local experts before taking action.

---

## Current Limitations

* CNN prediction depends on the quality and coverage of the trained dataset.
* RAG retrieval currently uses TF-IDF and keyword overlap rather than dense embeddings.
* Retrieved farmer advisory records may contain noisy, multilingual, or generic information.
* OpenAI advisory generation depends on the quality of retrieved evidence.
* Cloud deployment may require model files to be included or downloaded during setup.
* The system is an academic prototype, not a production agricultural diagnosis tool.

---

## Future Enhancements

* FAISS or ChromaDB vector search
* Sentence-transformer embeddings
* Multilingual translation support
* Larger crop image datasets
* More disease-specific expert knowledge base
* Cloud model hosting
* Mobile application interface
* Farmer voice assistant

---

## Project Type

This project is an academic AI prototype demonstrating multimodal agricultural intelligence using crop image classification, retrieval-augmented generation, OpenAI-based advisory generation, Grad-CAM explainability, backbone comparison, and a deployable FastAPI website.

````


