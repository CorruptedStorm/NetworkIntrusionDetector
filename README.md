# Network Intrusion Detector (Logistic Regression)

A machine learning model that classifies network connections as **normal** or **attack traffic**, using the NSL-KDD benchmark dataset.

## Why this project
Built to explore how machine learning can support network security monitoring — the same general idea behind commercial tools like intrusion detection systems (IDS) and endpoint security platforms, scaled down into a learning project.

## What it does
- Loads the NSL-KDD dataset (41 features per connection: protocol type, byte counts, error rates, etc.)
- Encodes categorical features and scales numeric ones
- Trains a logistic regression classifier to distinguish normal traffic from attacks
- Evaluates using accuracy, precision, recall, and a confusion matrix

## Results
- Accuracy: ~75%
- Precision: ~93%
- Recall: ~62%

Precision is strong (few false alarms), but recall is the weak point — the model misses a meaningful share of real attacks. That's a known tradeoff with a simple linear model and is the main direction for improvement.

## How to run
1. Open `network_intrusion_detector.ipynb` in Jupyter.
2. Run all cells top to bottom (the first cell downloads the dataset automatically).

## Next steps
- Compare against Random Forest / Gradient Boosting
- Classify specific attack types instead of binary normal/attack
- Tune the classification threshold to favor recall, since missed attacks are costlier than false alarms
- Feed in live-captured traffic (Wireshark/tshark) instead of a static dataset

## Tech
Python, pandas, scikit-learn, Jupyter
