# Stock Price Prediction using RNN-LSTM

A deep learning model that predicts Google stock prices using a Recurrent Neural Network (LSTM) architecture.

## Overview

This project implements a stock price prediction system using a stacked LSTM (Long Short-Term Memory) neural network. The model uses historical stock prices to predict future price movements of Google stock.

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
keras
tensorflow
```

## Project Structure

The project consists of three main parts:

## Data Preprocessing

- Imports Google stock price training data
- Applies MinMaxScaler for feature scaling (0-1 range)
- Creates a 60-day timestep structure for predictions
- Reshapes data into 3D format required for LSTM

## Model Architecture

The RNN model includes:
- 4 LSTM layers (50 units each)
- Dropout layers (20% dropout rate)
- Dense output layer
- Adam optimizer
- Mean squared error loss function

## Prediction and Visualization

- Processes test data using the same scaling
- Makes predictions on test set
- Visualizes results comparing actual vs predicted prices
## Key Features

**Data Structure**
- Uses 60 previous days for prediction
- 20 financial days per month
- 3-month prediction window

**Best Practices**
- Maintains consistent scaling between training and test data
- Uses proper data shapes for LSTM input
- Implements dropout for regularization

## Usage

1. Prepare your data:
   - Training data: CSV file with stock prices
   - Test data: CSV file with validation data

2. Run the model:
```python
# Train the model
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predicted_stock_price = regressor.predict(X_test)
```

3. Visualize results:
```python
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.show()
```

## Model Parameters

- LSTM Units: 50 per layer
- Dropout Rate: 0.2
- Batch Size: 32
- Epochs: 100

## Notes

- The model uses only the 'Open' price as a predictor
- Additional features (volume, close price, etc.) can be added for potentially better predictions
- The prediction window is optimized for 60 days of historical data