# Network Intrusion Detector

A machine learning pipeline that classifies network connections as **normal** or **attack traffic**, using the NSL-KDD benchmark dataset aswell as a more advanced version using the CIC-IDS2017 dataset.

## Why this project
Built to explore how machine learning can support network security monitoring — the same general idea behind commercial tools like intrusion detection systems (IDS) and endpoint security platforms, scaled down into a learning project.

## Structure
```
network-intrusion-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── pytest.ini
├── KDD/
│   ├── data/                                           # the kdd dataset
│   ├── notebooks/                                      # the kdd notebook
│   |   └── kdd_network_intrusion_detector.ipynb            # walkthrough version
│   └── src/                                            # source files used in the notebook
│       ├── kdd_data_loader.py                              # loading the kdd dataset
│       ├── preprocessing.py                                # encoding and scaling features
│       ├── model.py                                        # training and prediction
│       ├── neural_net.py                                   # training and prediction for nueral networks
│       ├── cross_validation.py                             # splits datasets for training and testing
│       └── evaluate.py                                     # metrics and reporting
└── CIC/
    ├── data/                                           # the cic dataset
    ├── notebooks/                                      # the cic notebook
    |   └── kdd_network_intrusion_detector.ipynb            # walkthrough version
    └── src/                                            # source files used in the notebook
        ├── cic_data_loader.py                              # loading the cic dataset
        ├── preprocessing.py                                # encoding and scaling features
        ├── model.py                                        # training and prediction
        ├── neural_net.py                                   # training and prediction for nueral networks
        ├── cross_validation.py                             # splits datasets for training and testing
        └── evaluate.py                                     # metrics and reporting
```

## How to run

**Run through python (this does not include neural network models):**
```
pip install -r requirements.txt
python main.py
```

**Notebook walkthrough:**
To view the NSL-KDD dataset models
```
jupyter notebook notebooks/kdd_network_intrusion_detector.ipynb
```
OR

To view the CIC-IDS2017 dataset models along with WIP WireShark application
```
jupyter notebook notebooks/cic_network_intrusion_detector.ipynb
```
