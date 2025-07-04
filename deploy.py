from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import os

app = FastAPI()

# Load models and scaler
lstm_model = load_model("ltm_model.h5")
svr_model = joblib.load("svr_model.pkl")
scaler = joblib.load("scaler.pkl")

# CSV file name
CSV_FILE = "battery_data.csv"

# Weighting factors
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
    time: float  # Elapsed time in seconds

@app.post("/predict")
async def predict(data: SensorData):
    try:
        # Prepare input for model
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

        # Standardization
        scaled_input = scaler.transform(input_sequence)
        sequence = np.array(scaled_input).reshape(1, 9, 1)  # 9 features

        lstm_pred = lstm_model.predict(sequence).flatten()[0]
        svr_pred = svr_model.predict([scaled_input[-1]])[0]

        final_pred = w1_opt * lstm_pred + w2_opt * svr_pred

        # Prepare new row for CSV
        new_row = {
            'cycle': data.cycle,
            'ambient_temperature': data.ambient_temperature,
            'capacity': data.capacity,
            'voltage_measured': data.voltage_measured,
            'current_measured': data.current_measured,
            'temperature_measured': data.temperature_measured,
            'current_load': data.current_load,
            'voltage_load': data.voltage_load,
            'time': data.time,
            'RUL': final_pred
        }

        # Check if CSV exists, if not create it
        if not os.path.isfile(CSV_FILE):
            df = pd.DataFrame(columns=[
                'cycle', 'ambient_temperature', 'capacity', 'voltage_measured',
                'current_measured', 'temperature_measured', 'current_load',
                'voltage_load', 'time', 'rul'
            ])
            df.to_csv(CSV_FILE, index=False)

        # Append to CSV safely
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        print(f"Logged Data: {new_row} | Predicted RUL: {final_pred}")

        return {"Estimated_RUL": final_pred}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
