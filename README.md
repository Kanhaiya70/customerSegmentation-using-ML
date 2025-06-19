# Customer Segmentation & Insights Dashboard

## Overview
A web-based ML-powered dashboard for customer segmentation, prediction, and reporting.

## Project Structure

```
customerSegmentation/
│
├── backend/                # Flask API & ML Model
│   ├── app.py              # Flask app with /predict endpoint
│   ├── segment_classifier.pkl # Trained ML model
│   ├── requirements.txt    # Backend dependencies
│   └── utils.py            # Helper functions (optional)
│
├── frontend/               # Streamlit Frontend
│   ├── app.py              # Streamlit app
│   ├── requirements.txt    # Frontend dependencies
│   └── assets/             # Images, logos, etc. (optional)
│
├── data/                   # Data for training/testing (optional)
│   └── customers.csv
│
└── README.md               # Project documentation
```

## Setup Instructions

### 1. Backend (Flask API)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 2. Frontend (Streamlit)
```bash
cd frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Features
- Customer input form
- ML-powered segment prediction
- Human-readable segment labels
- Visual insights (charts, graphs)
- PDF report generation

---

## API
- POST `/predict` (see PRD for details) 