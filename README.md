# 🌍 Climate Trend Forecasting with LSTM

## Project Overview
This project uses a **Recurrent Neural Network (RNN)**, specifically an **LSTM (Long Short-Term Memory)** model, to predict temperature trends based on historical climate data.  
Such a system can be part of **climate change modeling and mitigation**, enabling governments and scientists to plan for temperature shifts and their effects.

## Key Features
- Time series forecasting of temperature trends.
- Can be extended to include CO2, sea level, precipitation, etc.
- Interactive visualization of predictions vs. actual data.

## Requirements
- Python 3.x
- TensorFlow / Keras
- pandas
- numpy
- matplotlib
- scikit-learn

## Usage
1. Prepare climate data CSV (with `temperature` column).
2. Run `climate_forecast.py` to train and evaluate the model.
3. View the plot of predicted vs. actual temperature.

## Example Output
![Temperature Prediction](sample_output.png)

## Future Work
- Integrate multi-variate forecasting (more climate variables).
- Use external APIs for live climate data.
- Deploy as an interactive dashboard.
