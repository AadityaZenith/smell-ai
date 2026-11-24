"""Streamlit app: pick a smell, press Sniff, see the nose and the smell model work."""
import json

import joblib
import pandas as pd
import streamlit as st

from sensor import sniff, to_features
from smell_model import predict_smell

st.set_page_config(page_title="AI Nose", page_icon="👃")


@st.cache_resource
def load_nose():
    return joblib.load("models/nose.joblib")


@st.cache_data
def load_samples():
    with open("samples.json") as f:
        return json.load(f)


st.title("👃 Build an AI that can smell")
st.caption(
    "The nose here is simulated: 8 fake gas sensors. "
    "The smell model reads real molecule structures."
)

try:
    nose = load_nose()
except FileNotFoundError:
    st.error("Nose not trained yet. Run `python smell_model.py` and `python train_nose.py` first.")
    st.stop()

samples = load_samples()
choice = st.selectbox("Pick a smell to put under the nose", list(samples.keys()))

if st.button("Sniff", type="primary"):
    molecules = samples[choice]
    reading = sniff(molecules)
    proba = nose.predict_proba([to_features(reading)])[0]
    guess = nose.classes_[proba.argmax()]
    words = predict_smell(
        [m["smiles"] for m in molecules], [m["weight"] for m in molecules], top_k=5
    )

    st.subheader("1. Sensor channels")
    st.bar_chart(pd.DataFrame({"reading": reading},
                              index=[f"S{i + 1}" for i in range(len(reading))]))

    st.subheader("2. The nose's guess")
    st.metric("Looks like", guess, f"{proba.max():.0%} confident")
    st.bar_chart(pd.DataFrame({"probability": proba}, index=nose.classes_))

    st.subheader("3. The smell model describes it")
    st.write(", ".join(w for w, _ in words))

    with st.expander("Molecules in this sample"):
        st.table(pd.DataFrame(molecules)[["name", "smiles"]])
