#   Reccurent Neural Network


#   Part1- Data Preprocessing
#   Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

# Enable Metal GPU acceleration
tf.config.experimental.set_visible_devices(
    tf.config.list_physical_devices('GPU'), 'GPU'
)


#   importing training set
dataset_train =pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values    # : takes all the rows and 1:2 takes the second column only, 2 is excluded, 1  is the column index which is open price


#   Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1) )
training_set_scaled = sc.fit_transform(training_set)

#specific data structure that's most important step actually of data preprocessing  for RNN 
#create a data structure specfiying what RNN need to remember when predicting the next stock price
#and this is called number of time steps

#   Creating a data structure with 60 timesteps and 1 output
#60 timesteps means that at each time T the RNN is going to look at the 60 stock prices before time T,
#that is the stock prices between 60days before time T and time, based on the trends, it captures during these 60 previous timesteps
#it will try to predict the next output, that is the stock price at time T+1

#best timesteps ended up 60, and 60 is 60 previous financial days, since 20 days in one month, 60days corresponds three months.

X_train =[]
y_train = []

#1257 is the last number in training_set so we choose 1258
for i in range(60,1258 ):
    X_train.append(training_set_scaled[i-60:i,0]) # it takes 0 to 59.th element so T, 0 is column
    y_train.append(training_set_scaled[i,0]) #it predicts 60.th elementh so T+1
    
X_train, y_train = np.array(X_train), np.array(y_train)
    
#   Reshaping
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
# 1 is the number of predictors, we can add more predictors, for example volume, open price, close price etc.
# we can add more indicators to predict the stock price, but for now we only use one indicator, that is the stock price itself
    
#   Part 2 - Building the RNN

# Importing the Keras libraries and packages
# Use these instead
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Initializing the RNN

regressor = Sequential()

# Add the first LSTM layer and some Dropout regulariazation 

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
# units is the number of LSTM cells or LSTM neurons in the LSTM layer, 50 is a good number of neurons to start with 
# return_sequences is true because we are going to add another LSTM layer after this one
# input_shape is the shape of the input that we created in the previous step

regressor.add(Dropout(0.2)) # 0.2 is the rate of neurons that we want to drop out during the training, 
# 20% of the neurons of the LSTM layers will be ignored during the training, this will help to avoid overfitting
# since 20% of 50 is 10, 10 neurons will be ignored during the training at each iteration of the training 

# Adding a second LSTM layer and some Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer

regressor.add(Dense(units=1))
# dense function is used to add the output layer, units is the dimensionality of the output space, 
# 1 is the dimensionality of the output space because we are predicting a continous value, the stock price 


# Compiling the RNN 

regressor.compile(optimizer='adam', loss='mean_squared_error')

# adam is a very efficient stochastic gradient descent algorithm
# mean_squared_error is the loss function that is used for regression problems 



# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# batch_size is the number of samples per gradient update, 32 is a good number of batch_size to start with




#   Part3 - Making the predictions and visualizing the results

# Getting the real stock price of 2017

dataset_test =pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price= dataset_test.iloc[:,1:2].values    # : takes all the rows and 1:2 takes the second column only, 2 is excluded


# Getting the predicted stock price of 2017 
''' 
Core Requirements
- The model predicts stock prices based on 60 previous days of data
- Both training and test datasets are needed for predictions
- Proper data scaling is essential for accurate predictions
 
Important Technical Points 

    Data Structure Requirements
The input data must be in a 3D format for the RNN
The scaling must match the training data scaling
Only inputs should be scaled, not the test values 

    Timing Considerations
Each prediction requires 60 previous days of data
20 financial days constitute one month
The prediction window covers three months of historical data

    Best Practices
Never scale the actual test values
Use transform() instead of fit_transform() for consistent scaling
Maintain proper data shapes using reshape operations

 '''

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# axis = 0 means that the concatenation is vertical, axis = 1 means that the concatenation is horizontal
# we use the open stock prices to concatenate the dataset_train and dataset_test
# Open is the column that we want to concatenate 

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# len(dataset_total) - len(dataset_test) - 60: means that we take the last 60 stock prices of the training set, 
# and then we take the stock prices of the test set so that we can predict the first stock price of 2017 and then we concatenate the two dataframes 
# -60: means that we take the last 60 stock prices of the training set 
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # we only transform the inputs, we don't fit the inputs because we already fit the training set
X_test = [] 
for i in range(60, 80): # because we have 20 financial days in the test set 
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # we reshape the data to get the 3D structure that we need for the RNN
predicted_stock_price = regressor.predict(X_test) # we predict the stock prices of 2017 
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # we inverse the scaling because we want to get the real stock price values 





# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()