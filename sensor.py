"""A simulated array of 8 gas sensors (a fake electronic nose).

Each sensor responds differently to molecule properties: size, oil-loving-ness
(logP), oxygen, sulfur, nitrogen, aromatic rings and hydrogen-bond donors.
Readings get random noise, concentration changes and humidity drift.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
N_SENSORS = 8

# Fixed "sensitivity" of each sensor to each molecule property (seeded, so the
# same fake nose is rebuilt every time).
_rng = np.random.default_rng(42)
SENSITIVITY = _rng.normal(0, 1.0, size=(N_SENSORS, 7))
HUMIDITY_GAIN = _rng.uniform(0.2, 0.6, size=N_SENSORS)


def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array([
        Descriptors.MolWt(mol) / 100.0,
        Crippen.MolLogP(mol),
        sum(a.GetSymbol() == "O" for a in mol.GetAtoms()),
        sum(a.GetSymbol() == "S" for a in mol.GetAtoms()) * 2.0,
        sum(a.GetSymbol() == "N" for a in mol.GetAtoms()),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHBD(mol),
    ])


def clean_response(molecules):
    """Noise-free 8-channel response to a weighted mixture of molecules."""
    w = np.array([m["weight"] for m in molecules], dtype=float)
    w = w / w.sum()
    d = np.array([describe(m["smiles"]) for m in molecules])
    per_mol = np.log1p(np.exp(d @ SENSITIVITY.T))  # softplus keeps readings positive
    return (w[:, None] * per_mol).sum(axis=0)


def sniff(molecules, rng=None, noise=0.05, jitter=0.2):
    """One noisy reading: concentration change + humidity + sensor noise."""
    rng = rng or np.random.default_rng()
    jittered = [
        {**m, "weight": m["weight"] * rng.uniform(1 - jitter, 1 + jitter)}
        for m in molecules
    ]
    base = clean_response(jittered)
    concentration = rng.uniform(0.6, 1.4)
    humidity = rng.uniform(0.0, 1.0)
    reading = base * concentration * (1 + rng.normal(0, noise, N_SENSORS))
    reading += humidity * HUMIDITY_GAIN
    return reading


def to_features(reading):
    """Pattern of the 8 channels (divided by their sum) plus the log total."""
    total = reading.sum()
    return np.concatenate([reading / total, [np.log(total)]])
