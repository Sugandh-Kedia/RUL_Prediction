from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load models and scaler
lstm_model = load_model("ltm_model.h5")
svr_model = joblib.load("svr_model.pkl")
scaler = joblib.load("scaler.pkl")

SEQUENCE_LENGTH = 1
w1_opt = 0.958091373664059
w2_opt = 1 - w1_opt

class SensorData(BaseModel):
    cycle: float
    ambient_temperature: float
    capacity: float
    voltage_measured: float
    current_measured: float
    temperature_measured: float
    current_load: float
    voltage_load: float
    time: float

@app.post("/predict")
async def predict(data: SensorData):
    try:
        # Prepare input
        input_sequence = [[
            data.cycle,
            data.ambient_temperature,
            data.capacity,
            data.voltage_measured,
            data.current_measured,
            data.temperature_measured,
            data.current_load,
            data.voltage_load,
            data.time
        ]]

        scaled_input = scaler.transform(input_sequence)
        sequence = np.array(scaled_input).reshape(1, 9, 1)

        lstm_pred = lstm_model.predict(sequence).flatten()[0]
        svr_pred = svr_model.predict([scaled_input[-1]])[0]

        final_pred = w1_opt * lstm_pred + w2_opt * svr_pred

        return {"Estimated_RUL": final_pred}
    except Exception as e:
        return {"error": str(e)}
