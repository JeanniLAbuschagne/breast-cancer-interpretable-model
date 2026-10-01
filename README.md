# Interpretable Breast Cancer Prediction

A trustworthy, transparent decision-support model that classifies breast tumours as
benign or malignant from cell-nucleus measurements (Wisconsin Diagnostic dataset).

## Result
A **6-feature logistic regression** achieves a cross-validated **F1 = 0.961 [95% CI 0.955-0.966]**, AUC = 0.992 —
statistically on par with black-box ensembles while staying fully interpretable. Every prediction is
explained via odds ratios, and the decision threshold is selected on training data (0.40) to prioritise
catching cancers (sensitivity).

## Structure
- `data/raw/` — original `data.csv` (immutable; keep out of version control if sensitive)
- `src/pipeline.py` — end-to-end reproducible analysis
- `models/` — model parameters used by the GUI prototype
- `reports/figures/` — all generated charts
- `clinical_gui_prototype.html` — interactive "second-reader" GUI mock-up (open in a browser)
- `Breast_Cancer_Interpretable_Model_Report.docx` — full write-up

## Run
```bash
pip install -r requirements.txt
python src/pipeline.py
```

## Clinical note
A decision-support **second reader** — does not replace clinical judgement or confirmatory
pathology. Cases in the uncertain band (~0.20–0.60) are routed for expert review.
