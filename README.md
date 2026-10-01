# Network Intrusion Detector (Logistic Regression)

A machine learning pipeline that classifies network connections as **normal** or **attack traffic**, using the NSL-KDD benchmark dataset.

## Why this project
Built to explore how machine learning can support network security monitoring — the same general idea behind commercial tools like intrusion detection systems (IDS) and endpoint security platforms, scaled down into a learning project.

## Structure
```
network-intrusion-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── pytest.ini
├── main.py                 # runs the full pipeline end to end
├── data/                   # downloaded dataset (gitignored)
├── src/
│   ├── data_loader.py       # downloading and loading the dataset
│   ├── preprocessing.py     # encoding and scaling features
│   ├── model.py              # training and prediction
│   ├── neural_net.py         # training and prediction for nueral networks
│   └── evaluate.py           # metrics and reporting
└── notebooks/
    └── network_intrusion_detector.ipynb   # walkthrough version, imports from src/
```

## How to run

**Run through python (this does not include neural network models):**
```
pip install -r requirements.txt
python main.py
```

**Notebook walkthrough:**
```
jupyter notebook notebooks/network_intrusion_detector.ipynb
```

## Tech
Python, pandas, scikit-learn, Jupyter, pytest
