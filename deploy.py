import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# --- Load Models and Scalers ---
@st.cache_resource
def load_all_models():
    lstm_model = load_model("ltm_model.h5")
    svr_model = joblib.load("svr_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return lstm_model, svr_model, scaler

lstm_model, svr_model, scaler = load_all_models()

# --- Parameters ---
SEQUENCE_LENGTH = 1
w1_opt = 0.2958091373664059
w2_opt = 1 - w1_opt

# --- Streamlit UI ---
st.title("🔧 Predict RUL from Manual Input")

st.subheader("Enter Feature Values")

# Features
features = ["cycle", "ambient_temperature", "capacity", "voltage_measured", 
            "current_measured", "temperature_measured", "current_load", 
            "voltage_load", "time"]

# Build manual input for last SEQUENCE_LENGTH time steps
manual_input = []
for t in range(SEQUENCE_LENGTH):
    # st.markdown(f"#### ⏱ Time Step {t + 1}")
    step = []
    for feat in features:
        value = st.number_input(f"{feat} (t={t + 1})", key=f"{feat}_{t}")
        step.append(value)
    manual_input.append(step)

# --- Prediction Function ---
def predict_rul_manual(input_sequence, lstm_model, svr_model, scaler, sequence_length, weight_lstm, weight_svr):
    scaled_input = scaler.transform(input_sequence)

    # Reshape for LSTM: (1, 9, 1)
    sequence = np.array(scaled_input).reshape(1, 9, 1)

    lstm_pred = lstm_model.predict(sequence).flatten()[0]
    svr_pred = svr_model.predict([scaled_input[-1]])[0]

    final_pred = weight_lstm * lstm_pred + weight_svr * svr_pred
    return final_pred


# --- Predict Button ---
if st.button("🔍 Predict RUL"):
    try:
        # Flatten input to count non-zero values
        non_zero_count = np.count_nonzero(manual_input)

        if non_zero_count < 2:
            st.warning("⚠️ Please fill fields to get a valid prediction.")
        else:
            prediction = predict_rul_manual(manual_input, lstm_model, svr_model, scaler, SEQUENCE_LENGTH, w1_opt, w2_opt)
            st.success(f"📉 Estimated RUL: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"❌ Error: {e}")

