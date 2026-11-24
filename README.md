# Build an AI That Can Smell

A smell model that reads a molecule and predicts what it smells like, joined to a simulated electronic nose that tells smells apart. Runs on any laptop in a few minutes. No GPU, no hardware.

Built on the OpenPOM dataset: 4,983 molecules labelled with 138 smell words by perfume experts.

> The nose is **simulated** in code (`sensor.py`). Swap in a real BME688 gas sensor to go further.

## Run it (Python 3.10+)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smell_model.py              # downloads data, trains smell model (~1 min)
python train_nose.py               # trains the nose (~15 s)
streamlit run app.py               # opens the app
python test_app.py                 # optional: checks everything without a browser
```

## Files

| File | Purpose |
| --- | --- |
| `smell_model.py` | Morgan fingerprints + small neural network: molecule -> 138 smell words |
| `samples.json` | Six smells (coffee, vanilla candle, mint, lemon, banana, wood smoke) as lists of real aroma molecules |
| `sensor.py` | Simulated array of 8 gas sensors with noise and humidity drift |
| `train_nose.py` | Trains a random forest to name the smell from 8 sensor readings |
| `app.py` | Streamlit app |
| `test_app.py` | End-to-end check |

Typical results: smell model ~0.82 ROC-AUC on unseen molecules; nose ~99-100% on simulated sniffs (high because the simulator is also what it learned from).

## Use a real sensor

Wire a BME688 to an ESP32, step the heater through several temperatures, record each smell in its own jar over 3+ sessions on different days, then train the same random forest on those readings (divided by the clean-air reading) and test on a session it never saw. Avoid hot coffee (it mostly detects steam) and start with very different smells.

## Credits

Lee et al., *A principal odor map unifies diverse tasks in olfactory perception*, Science (2023). Dataset and model rebuild: OpenPOM (MIT), built on GoodScents and Leffingwell data.

## Author

Aaditya Kumar
