# Network Intrusion Detector


A machine learning pipeline that classifies network connections as **normal** or **attack traffic**. It uses the NSL-KDD benchmark dataset as a starting point, and the larger, more realistic CIC-IDS2017 dataset as a more advanced version.

## Why this project
Built to explore how machine learning can support network security monitoring: the same general idea behind commercial intrusion detection systems (IDS) and endpoint security platforms, scaled down into a learning project.

## Structure
```
network-intrusion-detector/
├── README.md
├── main.py                                         # run either dataset from the command line
├── requirements.txt
├── .gitignore
├── pytest.ini
├── results/                                        # charts and tables saved by main.py
├── KDD/
│   ├── data/                                       # put the NSL-KDD files here
│   ├── notebooks/
│   │   └── kdd_network_intrusion_detector.ipynb    # walkthrough version
│   └── src/                                        # source files used by the notebook and main.py
│       ├── kdd_data_loader.py                      # loading the NSL-KDD dataset
│       ├── preprocessing.py                        # encoding and scaling features
│       ├── model.py                                # logistic regression training and prediction
│       ├── neural_net.py                           # neural network training and prediction
│       ├── tree_models.py                          # gradient boosted trees (and an isolation forest)
│       ├── cross_validation.py                     # cross-validation helper
│       ├── compare.py                              # compares models and plots charts
│       └── evaluate.py                             # metrics and reporting
└── CIC/
    ├── data/                                       # put the CIC-IDS2017 CSV files here
    ├── notebooks/
    │   └── cic_network_intrusion_detector.ipynb    # walkthrough version
    └── src/                                        # same layout as KDD/src
        ├── cic_data_loader.py                      # loading the data and splitting it into train/test
        ├── preprocessing.py
        ├── model.py
        ├── neural_net.py
        ├── tree_models.py
        ├── cross_validation.py
        ├── compare.py
        └── evaluate.py
```

## Dataset Links
Neither dataset is included in the repo. Download them and place the files in the matching `data/` folder:

- **CIC-IDS2017**: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset (put the CSV files in `CIC/data/`)
- **NSL-KDD**: https://www.kaggle.com/datasets/hassan06/nslkdd (put `KDDTrain+.txt` and `KDDTest+.txt` in `KDD/data/`)

## How to run

**Run through python:**
```
pip install -r requirements.txt
python main.py
```

With no arguments, `main.py` asks which dataset to run. You can also choose directly:
```
python main.py --dataset kdd
python main.py --dataset cic
python main.py --dataset both
```

Models: logistic regression, a gradient boosted tree (HGB) and a neural net run by default. Other options:

| Option | What it does |
| --- | --- |
| `--models lr hgb hgb-deep mlp` | choose which models to train (`hgb-deep` is a deeper tree model) |
| `--cv kfold` | also run 5-fold cross-validation on the training set (`day` and `block` are CIC only) |
| `--split block` / `--split day` | CIC only: split inside each day (default), or hold out Friday |
| `--test-frac 0.1` | CIC block split only: fraction of blocks used for testing |
| `--sample 200000` | CIC only: use a random subset for a quick run |
| `--data-dir PATH` | use a different data folder (single dataset only) |

Charts and tables are saved in `results/`. The full CIC dataset is large, so the neural net and cross-validation can take a long time. Use `--sample` or `--models lr hgb` for a quick run.

### Notebook walkthroughs
NSL-KDD:
```
jupyter notebook KDD/notebooks/kdd_network_intrusion_detector.ipynb
```
CIC-IDS2017:
```
jupyter notebook CIC/notebooks/cic_network_intrusion_detector.ipynb
```

## How the CIC models are evaluated
The CIC data is split in three different ways, because each answers a different question:

- **Block split (default).** Each attack type on each day is cut into contiguous time blocks, and whole blocks go to either train or test, with a small gap between them. This avoids leaking near-identical neighboring flows. Every attack type appears in both sets, so this measures how well the models detect **known** attacks.
- **Plain k-fold cross-validation.** Rows are shuffled across folds. This tends to be optimistic, because similar flows from the same attack end up in both train and test.
- **Day-based cross-validation / Friday holdout (`--split day`, `--cv day`).** Whole days are held out, so the models are tested on attack types they have never seen. Recall drops sharply and varies a lot between folds. This measures how well the models detect **new** attacks.

In short, the models do very well on attacks they have seen before and much worse on attack types they haven't. See the notebook for the numbers and charts.


## Limitations
- CIC-IDS2017 is a lab-generated dataset from 2017, and published work found labeling and flow-generation issues in it. Near-perfect scores on it are common, so these results show the pipeline works, not that it would work on a real network.
- The models were trained and tested on one dataset. They have not been tested on traffic from other networks or captures.
- Some attack types are very rare (for example Heartbleed and Infiltration), so their per-attack detection rates are based on only a handful of test rows.

## Note
This main.py was created with Claude.
The majority of the underlying code was written by me while using Claude for minor editing.
