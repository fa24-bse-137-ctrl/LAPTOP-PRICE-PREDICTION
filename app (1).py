# ===============================================================
#  LAPTOP PRICE PREDICTION - STREAMLIT WEB APP
#  Needs these files in the same folder:
#     app.py, laptop_price_model_full_dataset.pkl, laptop_model_info.json,
#     requirements.txt
# ===============================================================
import json
import pickle
import os

import pandas as pd
import streamlit as st

MODEL_FILE = "laptop_price_model_full_dataset.pkl"
INFO_FILE = "laptop_model_info.json"

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")


# ---------- LOAD THE SAVED MODEL (only once) ----------
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_info():
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE) as f:
            return json.load(f)
    return {}


model = load_model()
info = load_info()

# The drop-down lists are taken from the model itself (the categories it learned),
# so no CSV file is needed on the server.
encoder = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
categorical_features = model.named_steps["preprocessor"].transformers_[1][2]
options = {name: sorted(values.tolist()) for name, values in zip(categorical_features, encoder.categories_)}


def pick(name, label, default):
    """Drop-down with a sensible default value."""
    values = options[name]
    index = values.index(default) if default in values else 0
    return st.selectbox(label, values, index=index)


# ---------- PAGE ----------
st.title("💻 Laptop Price Predictor")
st.write("Enter the laptop specifications and the model will predict its price.")

col1, col2 = st.columns(2)

with col1:
    company = pick("Company", "Company", "Dell")
    type_name = pick("TypeName", "Type", "Notebook")
    op_sys = pick("OpSys", "Operating system", "Windows 10")
    inches = st.number_input("Screen size (inches)", min_value=10.0, max_value=18.5, value=15.6, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

with col2:
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    ssd = st.selectbox("SSD (GB)", [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1024], index=8)
    hdd = st.selectbox("HDD (GB)", [0, 32, 128, 500, 508, 1024, 2048], index=0)

screen = pick("ScreenResolution", "Screen resolution", "Full HD 1920x1080")
cpu = pick("Cpu", "Processor (CPU)", "Intel Core i5 7200U 2.5GHz")
gpu = pick("Gpu", "Graphics card (GPU)", "Intel HD Graphics 620")

if st.button("Predict price", type="primary"):
    # same columns, same order as in the training notebook
    laptop = pd.DataFrame([{
        "Inches": inches, "Ram": ram, "Weight": weight,
        "SSD_GB": ssd, "HDD_GB": hdd,
        "Company": company, "TypeName": type_name,
        "ScreenResolution": screen, "Cpu": cpu, "Gpu": gpu, "OpSys": op_sys,
    }])
    price = model.predict(laptop)[0]

    st.success(f"Predicted price: **{price:,.0f}**")
    if "cv_rmse" in info:
        st.caption(
            f"Typical error of the model ({info['best_algorithm']}, 10-fold cross validation): "
            f"about ±{info['cv_rmse']:,.0f}"
        )

with st.expander("About this app"):
    st.write(
        "The model was trained on the laptop price dataset and selected from 5 regression "
        "algorithms (Ridge, SVR, Extra Trees, AdaBoost, Hist Gradient Boosting) using "
        "10-fold cross validation."
    )
    if info:
        st.write(f"Best algorithm: **{info.get('best_algorithm')}**")
