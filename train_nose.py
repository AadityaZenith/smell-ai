"""Train the nose: 8 sensor readings -> which smell it is."""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sensor import sniff, to_features

N_SNIFFS = 200
NOSE_PATH = os.path.join("models", "nose.joblib")


def main():
    os.makedirs("models", exist_ok=True)
    with open("samples.json") as f:
        samples = json.load(f)

    rng = np.random.default_rng(0)
    X, y = [], []
    for name, molecules in samples.items():
        for _ in range(N_SNIFFS):
            X.append(to_features(sniff(molecules, rng)))
            y.append(name)
    X, y = np.array(X), np.array(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"Nose accuracy on new sniffs: {acc:.1%}")
    joblib.dump(clf, NOSE_PATH)
    print(f"Saved {NOSE_PATH}")


if __name__ == "__main__":
    main()
