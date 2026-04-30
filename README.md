# AgriAssist: Multimodal Crop Disease Diagnosis and Localized Advisory System

## Milestone 1 Required Deliverables
- Domain research note submitted: `domain_note.md`
- Data pipeline working and data loaded: `src/02_data_pipeline.py`
- Initial model running with preliminary results: `src/03_train_initial_model.py`

## Completed Results Included
The `results/` folder already contains completed Milestone 1 outputs.

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/01_create_sample_dataset.py
python3 src/02_data_pipeline.py
python3 src/03_train_initial_model.py
python3 src/04_rag_advisory_demo.py
python3 src/05_synthetic_data_demo.py
```
