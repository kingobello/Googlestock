<h3>GOOGL Stock Price Prediction using LSTM</h3>


Overview

This project leverages Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to predict Google (GOOGL) stock prices based on historical data. The project includes data preprocessing, model training, evaluation, and visualization of predictions against actual prices.

Features:

Fetches historical stock data using the Yahoo Finance API.

Scales and preprocesses the Close price for LSTM model training.

Builds a robust LSTM-based model for time-series forecasting.

Evaluates the model using metrics like:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R² Score

Visualizes the model's performance with an interactive plot comparing actual and predicted prices.
Includes functionality to save and reload the trained model.
Installation

Prerequisites
Python 3.9 or later
TensorFlow/Keras
Scikit-learn
Matplotlib
Yahoo Finance API (via yfinance)
Setup
Clone the repository:
git clone https://github.com/kingobello/googlestock.git
cd googlestock
Install dependencies:
pip install -r requirements.txt
Usage

Run the notebook or script:
python googlestock.py
Or open the Jupyter Notebook (googlestock.ipynb) for a step-by-step workflow.
The script will:
Download historical GOOGL stock data (2015–2024).
Preprocess and train the LSTM model.
Display performance metrics.
Plot actual vs. predicted stock prices.
To save the trained model:
model.save('googl_stock_predictor.h5')
To reload the model for future predictions:
from keras.models import load_model
model = load_model('googl_stock_predictor.h5')
Results

MSE: 13.358
MAE: 2.801
R² Score: 0.981
The model provides highly accurate predictions of GOOGL stock prices.

Future Work

Incorporate additional features like trading volume, moving averages, or volatility.
Experiment with other architectures (e.g., GRU, Transformer).
Perform hyperparameter tuning for improved performance.
Extend to a larger dataset for better generalization.
License

This project is licensed under the MIT License.


Yahoo Finance API
TensorFlow/Keras
Scikit-learn
