<div align="center">

# 👃 Smell AI

### Building an AI system that understands and identifies smells

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenPOM](https://img.shields.io/badge/Dataset-OpenPOM-8A2BE2)](https://github.com/BioMachineLearning/openpom)

**Translate molecular structures into smell descriptors — and recognize smells with a simulated eight-channel electronic nose.**

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [Demo](#-demo) • [Project Structure](#-project-structure) • [Roadmap](#-roadmap)

</div>

---

## 🌟 Overview

**Smell AI** is an experimental Python project that combines molecular informatics with machine learning to explore one of the senses that has traditionally been difficult to encode digitally.

The project has two complementary parts:

1. **A molecule-to-smell model** — converts a molecule's SMILES structure into a **2048-bit Morgan fingerprint**, then predicts up to 138 expert-defined smell descriptors.
2. **A simulated electronic nose** — creates noisy readings from an eight-sensor array and uses a Random Forest classifier to identify one of six aroma profiles.

Everything is designed to run on a regular laptop. No GPU and no physical hardware are required for the default demo.

> [!IMPORTANT]
> The electronic nose is **fully simulated** in `sensor.py`. Sensor readings are generated from molecular properties and noise functions; they are not measurements from a real BME688 or other gas sensor.

---

## ✨ Features

- 🧪 **138 smell descriptors** trained on curated OpenPOM fragrance data
- 🧬 **SMILES-based molecular prediction** using RDKit Morgan fingerprints
- 👃 **Eight-channel simulated sensor array** with noise, concentration, and humidity effects
- 🤖 **Random Forest smell classifier** trained from simulated sniff samples
- 📊 **Interactive Streamlit interface** with sensor charts and confidence scores
- ☕ **Six ready-made aroma profiles:** coffee, vanilla candle, mint, lemon, banana, and wood smoke
- 🧱 **Simple modular structure** that is easy to extend or replace with real sensor data
- 💻 **CPU-friendly**, portable, and suitable for experimentation or learning

---

## 🖼️ Demo

The Streamlit app lets you:

1. Choose an aroma profile.
2. Generate a simulated sensor reading.
3. View the eight-channel response.
4. See the nose's prediction and confidence.
5. Display smell words predicted from the real molecular structures.

Launch it with:

```bash
streamlit run app.py
```

By default, the interface appears at `http://localhost:8501`.

---

## 🚀 Quick Start

### Requirements

- Python **3.10 or newer**
- `pip`
- Internet access on the first model run to download the OpenPOM dataset
- Approximately 1–2 GB of free disk space is recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/AadityaZenith/smell-ai.git
cd smell-ai

# Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Train the models

```bash
python smell_model.py
python train_nose.py
```

The first command downloads **4,983 OpenPOM molecules**, trains the smell model, and saves:

```text
models/smell_model.joblib
```

The second command creates 1,200 simulated sniffs and saves:

```text
models/nose.joblib
```

### Run the application

```bash
streamlit run app.py
```

### Run the end-to-end check

```bash
python test_app.py
```

This verifies model predictions, simulated nose accuracy, and that the Streamlit application can run and respond without opening a browser.

---

## 🧠 How It Works

```text
SMILES Molecules
      │
      ▼
Morgan Fingerprints ──► Neural Network ──► 138 smell descriptors

Aroma Profile
      │
      ▼
Molecular Properties ──► 8-Sensor Simulator ──► Random Forest ──► Aroma Guess
```

### 1. Molecule → smell descriptors

`smell_model.py` uses RDKit to generate a **2048-bit Morgan fingerprint** for each molecule. The fingerprint describes local molecular substructures and is passed to a small multilayer perceptron:

```text
2048 fingerprint bits
        │
        ▼
512-unit hidden layer
        │
        ▼
138 smell labels
```

Multiple molecules can be combined using their configured weights. For example, a coffee profile contains molecules such as furfuryl mercaptan, guaiacol, furfural, and maltol.

### 2. Molecular properties → sensor readings

`sensor.py` calculates seven properties for every molecule:

- Molecular weight
- Lipophilicity (`logP`)
- Oxygen atoms
- Sulfur atoms
- Nitrogen atoms
- Aromatic rings
- Hydrogen-bond donors

Each of the eight virtual sensors is assigned fixed sensitivity values. The model adds:

- concentration variation
- humidity response
- Gaussian sensor noise
- random molecule-weight jitter

### 3. Sensor reading → aroma classification

`train_nose.py` generates hundreds of simulated sniffs for each aroma and extracts nine features:

- Eight normalized sensor channels
- One logarithmic total-response feature

A **200-tree Random Forest** learns to classify these readings as coffee, vanilla candle, mint, lemon, banana, or wood smoke.

---

## 📊 Expected Results

| Component | Typical result | Notes |
|---|---:|---|
| Molecule model | ~0.82 mean ROC-AUC | Evaluated on held-out molecules |
| Simulated nose | ~99–100% accuracy | Generated from the same simulator used to train it |
| End-to-end check | Nose accuracy > 80% | Configured in `test_app.py` |

> [!NOTE]
> The simulated nose's high accuracy primarily demonstrates that a learned classifier can distinguish patterns generated by the simulator. It does **not** establish real-world sensor accuracy or human smell equivalence.

---

## 🗂️ Project Structure

```text
smell-ai/
├── app.py             # Streamlit interactive application
├── smell_model.py     # OpenPOM training and molecule → smell prediction
├── sensor.py          # Eight-channel simulated electronic nose
├── train_nose.py      # Random Forest training for simulated sniffs
├── test_app.py        # End-to-end smoke test
├── samples.json       # Aroma profiles and real aroma-molecule SMILES
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT license
└── README.md          # Project documentation
```

Runtime-generated folders:

```text
data/                  # Downloaded OpenPOM CSV
models/                # Trained model artifacts
  ├── smell_model.joblib
  └── nose.joblib
```

---

## 🔧 Configuration

### Molecule model

The core settings are in `smell_model.py`:

| Setting | Default | Purpose |
|---|---:|---|
| `N_BITS` | 2048 | Morgan fingerprint size |
| Hidden layer | 512 | Neural network capacity |
| Maximum iterations | 40 | MLP training limit |
| Test split | 20% | Held-out evaluation data |
| Top predictions | 5 | Number of displayed smell words |

### Electronic nose

The core settings are in `sensor.py` and `train_nose.py`:

| Setting | Default | Purpose |
|---|---:|---|
| `N_SENSORS` | 8 | Number of virtual sensor channels |
| Noise | 5% | Gaussian sensor noise |
| Jitter | 20% | Random variation in molecule weights |
| `N_SNIFFS` | 200 per smell | Training samples per aroma |
| Forest size | 200 trees | Random Forest model size |

---

## 🌐 Using a Real Sensor

The simulated sensor layer can eventually be replaced with real measurements from a BME688 or another gas sensor array.

A practical extension would involve:

1. Recording multiple sessions of every target smell.
2. Testing on multiple days to capture drift and environmental changes.
3. Keeping sensor temperature, humidity, and exposure time consistent.
4. Including an empty-jar baseline and unknown-smell samples.
5. Creating an independent test set across days and sessions.
6. Feeding real readings into the existing feature format.
7. Retraining the Random Forest classifier.

The six sample profiles are molecule libraries, not promises that a real device will identify a physical coffee, vanilla candle, mint, lemon, banana, or smoke sample.

---

## 🛣️ Roadmap

- [ ] Add automatic model and artifact validation
- [ ] Add unit tests for molecular parsing and sensor feature generation
- [ ] Build a Docker or dev-container setup
- [ ] Add custom aroma-profile creation in the UI
- [ ] Add SHAP or feature-importance explanations
- [ ] Support reading and exporting SMILES mixtures
- [ ] Add real BME688/BME680 sensor integration
- [ ] Support drift correction and calibration workflows
- [ ] Add model benchmark exports
- [ ] Add CI workflows for tests and dependency checks

---

## ⚠️ Limitations

- Olfaction is highly subjective and strongly influenced by genetics and context.
- A fingerprint is a chemical representation, not a complete physical odor model.
- OpenPOM labels do not represent every aroma or individual perception.
- Simulated sensor values should not be interpreted as real gas concentrations.
- The virtual nose can be memorized easily because its training and test data come from the same simulator.
- Model performance will vary with molecule validity, data quality, and future library updates.

This project is intended for learning, experimentation, and prototyping—not medical, safety, fragrance-compliance, or industrial odor-detection decisions.

---

## 🤝 Contributing

Contributions are welcome. Ideas include:

- Improving molecular representations
- Adding aroma datasets
- Implementing real sensor support
- Improving validation and test coverage
- Designing clearer visualizations
- Extending the UI and documentation

```bash
# Create a feature branch
git checkout -b feature/my-improvement

# Commit and push your changes
git add .
git commit -m "Add my improvement"
git push origin feature/my-improvement
```

Then open a pull request on GitHub.

---

## 📚 Data and References

The smell-model dataset is based on the **OpenPOM** community project:

- **OpenPOM repository:** https://github.com/BioMachineLearning/openpom
- **Dataset:** `curated_GS_LF_merged_4983.csv`
- **Source datasets:** GoodScents and Leffingwell
- **License:** MIT

The OpenPOM project builds on the Principal Odor Map work from:

> Lee, Brian K. et al. *“A principal odor map unifies diverse tasks in olfactory perception.”* **Science**, 2023.

If you use this project in academic work, please cite both the relevant Science publication and the OpenPOM repository/data source you use.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Aaditya Kumar**  
GitHub: [@AadityaZenith](https://github.com/AadityaZenith)

<div align="center">

### ⭐ If you find this project interesting, consider starring the repository!

</div>
