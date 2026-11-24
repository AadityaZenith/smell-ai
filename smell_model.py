"""Smell model: molecule (SMILES) -> smell words.

Run `python smell_model.py` to download the OpenPOM dataset and train the model.
Other files import `predict_smell()` from here.
"""
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

DATA_URL = (
    "https://raw.githubusercontent.com/BioMachineLearning/openpom/main/"
    "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
)
DATA_PATH = os.path.join("data", "openpom_4983.csv")
MODEL_PATH = os.path.join("models", "smell_model.joblib")
N_BITS = 2048


def fingerprint(smiles):
    """Morgan fingerprint: a bit vector of the molecule's chemical fragments."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)
    return np.array(list(fp), dtype=np.uint8)


def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading OpenPOM dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def train():
    os.makedirs("models", exist_ok=True)
    df = download_data()
    labels = [c for c in df.columns if c not in ("nonStereoSMILES", "descriptors")]

    X, rows = [], []
    for i, smi in enumerate(df["nonStereoSMILES"]):
        fp = fingerprint(smi)
        if fp is not None:
            X.append(fp)
            rows.append(i)
    X = np.array(X)
    Y = df.loc[rows, labels].values.astype(int)
    print(f"{len(X)} molecules, {len(labels)} smell words")

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=0)
    # A small neural network: 2048 fingerprint bits -> 512 hidden units -> 138 smell words.
    # (A random forest also works but its saved file is ~2 GB for 138 labels.)
    clf = MLPClassifier(
        hidden_layer_sizes=(512,), alpha=1e-3, max_iter=40,
        batch_size=64, random_state=0,
    )
    clf.fit(X_tr, Y_tr)

    proba = clf.predict_proba(X_te)
    aucs = []
    for j in range(len(labels)):
        if 0 < Y_te[:, j].sum() < len(Y_te):
            aucs.append(roc_auc_score(Y_te[:, j], proba[:, j]))
    print(f"ROC-AUC on unseen molecules: {np.mean(aucs):.2f}  (0.5 = guessing, 1 = perfect)")

    # refit on everything for the final model
    clf.fit(X, Y)
    joblib.dump({"model": clf, "labels": labels}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


_cache = {}


def _load():
    if "bundle" not in _cache:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Run `python smell_model.py` first.")
        _cache["bundle"] = joblib.load(MODEL_PATH)
    return _cache["bundle"]


def predict_smell(smiles_list, weights=None, top_k=5):
    """Predict smell words for one molecule or a weighted mixture of molecules."""
    bundle = _load()
    clf, labels = bundle["model"], bundle["labels"]
    fps, w = [], []
    for i, smi in enumerate(smiles_list):
        fp = fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            w.append(1.0 if weights is None else weights[i])
    if not fps:
        return []
    probs = clf.predict_proba(np.array(fps))
    mix = np.average(probs, axis=0, weights=w)
    order = np.argsort(mix)[::-1][:top_k]
    return [(labels[j], float(mix[j])) for j in order]


if __name__ == "__main__":
    train()
