"""
main.py - run the network intrusion detection experiments from the command line.

Examples:
    python main.py                                  # asks which dataset to run
    python main.py --dataset kdd
    python main.py --dataset cic
    python main.py --dataset both
    python main.py --dataset cic --models lr hgb mlp --cv kfold
    python main.py --dataset cic --split day        # hold out Friday (unseen attack types)
    python main.py --dataset cic --sample 200000    # quick run on a random subset

Charts and tables are saved to the results/ folder.

Note: This file was created with Claude, but the majority of the underlying code was written by me while using Claude for minor editing.
"""

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
# A model that flags nothing in a CV fold gets precision 0 (sklearn warns about it)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

MODEL_CHOICES = ("lr", "hgb", "hgb-deep", "mlp")
DEFAULT_MODELS = ("lr", "hgb")



# Loading the right `src` package
def load_package(folder, modules):
    """Import the `src` package that lives in CIC/ or KDD/.

    Both folders contain a package called `src`, so Python would mix them up
    if both were imported in one process. We clear the cached `src` modules
    and point sys.path at the requested folder before importing.
    """
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if p not in (str(ROOT / "CIC"), str(ROOT / "KDD"))]
    sys.path.insert(0, str(ROOT / folder))
    return SimpleNamespace(**{m: importlib.import_module(f"src.{m}") for m in modules})


# Shared steps
def build_models(pkg, names, X_train, y_train, train_lr):
    models = {}
    for key in names:
        start = time.time()
        if key == "lr":
            label, model = "Logistic Regression", train_lr(X_train, y_train)
        elif key == "hgb":
            label, model = "HGB (default)", pkg.tree_models.train_hgb(X_train, y_train)
        elif key == "hgb-deep":
            label = "HGB (depth 5, 200 iter)"
            model = pkg.tree_models.train_hgb(X_train, y_train, max_depth=5, max_iter=200)
        else:  # mlp
            label = "Neural Net (32-16)"
            model = pkg.neural_net.train_mlp(X_train, y_train, hidden_layer_size=(32, 16))
        models[label] = model
        print(f"  trained {label} in {time.time() - start:.0f}s")
    return models


def report(pkg, models, X_test, y_test, attack_labels, prefix, title):
    """Print metrics, save charts, and print per-attack-type detection rates."""
    RESULTS.mkdir(exist_ok=True)

    results_df, cms = pkg.compare.compare_models(models, X_test, y_test)
    print("\n" + results_df.round(4).to_string())
    for name, cm in cms.items():
        print(f"  {name}: missed attacks (FN) = {cm[1, 0]:,}, false alarms (FP) = {cm[0, 1]:,}")

    pkg.compare.plot_bar_chart(results_df, title=title, path=str(RESULTS / f"{prefix}_comparison.png"))
    plt.close("all")
    pkg.compare.plot_confusion_matrices(cms, path=str(RESULTS / f"{prefix}_confusion_matrices.png"))
    plt.close("all")

    # share of rows flagged as an attack, per true label, for every model
    rates = {}
    for name, model in models.items():
        per = pkg.evaluate.per_attack_recall(attack_labels, model.predict(X_test))
        rates[name] = per["rate"]
        rows = per["rows"]
    table = pd.DataFrame(rates).assign(rows=rows)
    table = table.sort_values(table.columns[0])
    print("\nShare of rows flagged as attack, by true label")
    print("(for the benign/normal row this is the false alarm rate, lower is better):")
    print(table.round(3).to_string())
    table.to_csv(RESULTS / f"{prefix}_per_attack.csv")


def cross_validate_models(pkg, models, X_train, y_train, cv, groups, prefix, title):
    print(f"\nCross-validation ({title}) - this refits every model for every fold, so it can be slow.")
    means, stds, cms = pkg.cross_validation.run_cv(models, X_train, y_train, cv=cv, groups=groups)
    print("\n" + means.round(4).to_string())
    pkg.compare.plot_bar_chart(means, title=title, errors=stds, path=str(RESULTS / f"{prefix}_cv.png"))
    plt.close("all")
    pkg.compare.plot_confusion_matrices(cms, path=str(RESULTS / f"{prefix}_cv_confusion_matrices.png"))
    plt.close("all")



# CIC-IDS2017
def run_cic(args):
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    pkg = load_package("CIC", ["cic_data_loader", "preprocessing", "model", "evaluate",
                               "neural_net", "compare", "tree_models", "cross_validation"])

    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "CIC" / "data"
    print(f"Loading CIC-IDS2017 from {data_dir} ...")
    df = pkg.cic_data_loader.cic_load_data(data_dir=str(data_dir))
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).sort_index()  # keep time order
        print(f"Using a random sample of {len(df):,} rows")

    if args.split == "block":
        train, test = pkg.cic_data_loader.split_within_days(df, test_frac=args.test_frac)
        split_note = "within-day block split (known attack types)"
    else:
        train, test = pkg.cic_data_loader.split_by_day(df)
        split_note = "Friday held out (unseen attack types)"
    print(f"Split: {split_note}\nTrain rows: {len(train):,}  Test rows: {len(test):,}")

    X_train, X_test, y_train, y_test, _ = pkg.preprocessing.prepare_cic_features(
        train, test, drop_cols=("Label", "Day", "is_attack", "Block"))

    print("Training models ...")
    models = build_models(pkg, args.models, X_train, y_train, pkg.model.train_lr_model)
    report(pkg, models, X_test, y_test, test["Label"], f"cic_{args.split}",
           f"CIC-IDS2017 - {split_note}")

    if args.cv == "kfold":
        cv, groups = StratifiedKFold(5, shuffle=True, random_state=42), None
        cv_title = "CIC k-fold CV"
    elif args.cv == "day":
        groups = train["Day"].replace({"monday": "tuesday"})  # Monday is benign-only
        cv, cv_title = GroupKFold(n_splits=groups.nunique()), "CIC day-based CV (unseen attack types)"
    elif args.cv == "block":
        if "Block" not in train:
            sys.exit("--cv block needs --split block")
        groups = train["Block"]
        cv, cv_title = GroupKFold(n_splits=5), "CIC block-based CV"
    else:
        return
    cross_validate_models(pkg, models, X_train, y_train, cv, groups, f"cic_{args.cv}", cv_title)


# NSL-KDD
def run_kdd(args):
    from sklearn.model_selection import StratifiedKFold

    if args.cv in ("day", "block"):
        sys.exit("--cv day/block only apply to the CIC dataset (NSL-KDD has no day column). Use --cv kfold.")

    pkg = load_package("KDD", ["kdd_data_loader", "preprocessing", "model", "evaluate",
                               "neural_net", "compare", "tree_models", "cross_validation"])

    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "KDD" / "data"
    print(f"Loading NSL-KDD from {data_dir} ...")
    train, test = pkg.kdd_data_loader.kdd_load_data(data_dir=str(data_dir))
    print(f"Train rows: {len(train):,}  Test rows: {len(test):,}")

    X_train, X_test, y_train, y_test, _ = pkg.preprocessing.prepare_kdd_features(train, test)

    print("Training models ...")
    models = build_models(pkg, args.models, X_train, y_train, pkg.model.train_model)
    report(pkg, models, X_test, y_test, test["label"], "kdd", "NSL-KDD - official train/test split")

    if args.cv == "kfold":
        cross_validate_models(pkg, models, X_train, y_train,
                              StratifiedKFold(5, shuffle=True, random_state=42), None,
                              "kdd_kfold", "NSL-KDD k-fold CV")


# Command line
def parse_args():
    p = argparse.ArgumentParser(description="Run the network intrusion detection experiments.")
    p.add_argument("--dataset", choices=["cic", "kdd", "both"],
                   help="which dataset to run (asked interactively if omitted)")
    p.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS),
                   help="models to train (default: lr hgb). mlp is slow on CIC.")
    p.add_argument("--cv", choices=["none", "kfold", "day", "block"], default="none",
                   help="also run cross-validation on the training set (day/block: CIC only)")
    p.add_argument("--split", choices=["block", "day"], default="block",
                   help="CIC only. block = split inside each day (known attacks); "
                        "day = hold out Friday (unseen attacks)")
    p.add_argument("--test-frac", type=float, default=0.10,
                   help="CIC block split only: fraction of blocks used for testing (default 0.10)")
    p.add_argument("--sample", type=int, default=0,
                   help="CIC only: use a random sample of this many rows for a quick run")
    p.add_argument("--data-dir", help="override the data folder (only valid with a single dataset)")
    args = p.parse_args()

    if args.dataset is None:
        if not sys.stdin.isatty():
            p.error("--dataset is required when not running interactively")
        choice = input("Which dataset would you like to run? [cic / kdd / both]: ").strip().lower()
        if choice not in ("cic", "kdd", "both"):
            p.error(f"unknown dataset '{choice}'")
        args.dataset = choice
    if args.data_dir and args.dataset == "both":
        p.error("--data-dir can't be used with --dataset both")
    return args


def main():
    args = parse_args()
    runners = {"cic": run_cic, "kdd": run_kdd}
    for name in (["cic", "kdd"] if args.dataset == "both" else [args.dataset]):
        print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
        start = time.time()
        try:
            runners[name](args)
        except FileNotFoundError as err:
            folder = "CIC" if name == "cic" else "KDD"
            print(f"\nCouldn't find the {name.upper()} data ({err}).\n"
                  f"Put the dataset files in {ROOT / folder / 'data'} (or pass --data-dir).")
            sys.exit(1)
        print(f"\n{name.upper()} finished in {time.time() - start:.0f}s. Results saved in {RESULTS}")


if __name__ == "__main__":
    main()