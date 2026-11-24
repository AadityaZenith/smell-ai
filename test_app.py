"""Quick check that everything works, without opening a browser."""
import json

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

from sensor import sniff, to_features
from smell_model import predict_smell

with open("samples.json") as f:
    samples = json.load(f)
nose = joblib.load("models/nose.joblib")
rng = np.random.default_rng(123)

print("Smell model:")
for name, mols in samples.items():
    words = predict_smell([m["smiles"] for m in mols], [m["weight"] for m in mols])
    assert words, f"no words for {name}"
    print(f"  {name}: {', '.join(w for w, _ in words)}")

print("Nose:")
correct = total = 0
for name, mols in samples.items():
    for _ in range(20):
        correct += nose.predict([to_features(sniff(mols, rng))])[0] == name
        total += 1
print(f"  {correct}/{total} correct")
assert correct / total > 0.8

print("App:")
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
at.button[0].click().run()
assert not at.exception, at.exception
print("  app runs and Sniff works")
print("All good.")
