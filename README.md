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
│   └── evaluate.py           # metrics and reporting
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_evaluate.py
└── notebooks/
    └── network_intrusion_detector.ipynb   # walkthrough version, imports from src/
```

## How to run

**Full pipeline:**
```
pip install -r requirements.txt
python main.py
```

**Notebook walkthrough:**
```
jupyter notebook notebooks/network_intrusion_detector.ipynb
```

**Tests:**
```
pytest
```

## Results
- Accuracy: ~75%
- Precision: ~93%
- Recall: ~62%

Precision is strong (few false alarms), but recall is the weak point — the model misses a meaningful share of real attacks. That's the main direction for improvement.

## Next steps
- Compare against Random Forest / Gradient Boosting
- Classify specific attack types instead of binary normal/attack
- Tune the classification threshold to favor recall, since missed attacks are costlier than false alarms
- Feed in live-captured traffic (Wireshark/tshark) instead of a static dataset

## Tech
Python, pandas, scikit-learn, Jupyter, pytest
