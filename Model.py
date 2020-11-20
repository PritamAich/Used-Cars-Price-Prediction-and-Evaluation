#This is the end file of the ML model for the Used cars price prediction 

#This model is considered to be performing best among five other ML models tested on this dataset.

#To recreate the model, run this file in the terminal and the code will do the rest.

#Note: This program will use the "used_cars_price_cleaned.csv" as the dataset since the file is already preprocessed and all necessary
#Feature engineering tasks are done prior

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_log_error
import pickle

import warnings
warnings.filterwarnings("ignore")

#Reading the file
raw_data = pd.read_csv('used_cars_price_cleaned.csv')

data = raw_data.copy()
data = data.drop('Unnamed: 0', axis = 1)


#Splitting of dataset
from sklearn.model_selection import train_test_split

x = data.drop('priceUSD', axis = 1) #Independent variables
y = data['priceUSD'] #Dependent variables

#Training and test data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,random_state = 42)

#Training and validation data split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)

print("Training, Validation and test sets created.")
print("Model will be trained on Training set and evaluated on validation set.")
print("After evaluation on validation set, an RMLE(Root mean squared log error) score and Accuracy(R^2) score will be displayed for the validation set.")

print("-----------------------------------------------------------------------------------------------------------")


#Scaling the data
#Since range of values for each feature is different, they need to be scaled with a fixed range
#MinMaxScaler scales features within a range of (0,1)
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
  
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

#Since the Price of cars is not normally distributed, skewed to the left, hence it will be best to perfrom a log transformation on price.
y_train_log = np.log(y_train)

y_val_log = np.log(y_val)

y_test_log = np.log(y_test)


#CatBoost model creation
from catboost import CatBoostRegressor

cb = CatBoostRegressor(bagging_temperature = 3, depth = 8, l2_leaf_reg = 0.5, learning_rate = 0.05)

cb.fit(x_train_scaled, y_train_log, verbose = 200) #Fitting or training the data

y_val_pred_log = cb.predict(x_val_scaled) #Predicting on validation set

rmsle = np.sqrt(mean_squared_log_error(y_val_log, y_val_pred_log)) #Evaluating model rmsle score by comparing actual and predicted results
score = ((cb.score(x_val_scaled,y_val_log))*100).round(3)

print("RMSLE score for validation set: ", rmsle)

print("Accuracy score for validation set: ", score)

print("-----------------------------------------------------------------------------------------------------------")


print("The model will now be tested on Test set and and RMSlE score and Accuracy will be displayed for test set.")


#Testing model on test set
y_test_pred_log = cb.predict(x_test_scaled) #Predicting on test set

rmsle = np.sqrt(mean_squared_log_error(y_test_log, y_test_pred_log)) #Evaluating model rmsle score by comparing actual and predicted results
score = ((cb.score(x_test_scaled,y_test_log))*100).round(3)

print("RMSLE score for test set: ", rmsle)

print("Accuracy score for test set: ", score)

scaler_auth = input("Do you want to save the scaler object?(Y/N)")

if scaler_auth == "Y" or scaler_auth == "y":
    #Saving Scaler object
    filehandler = open('scaler.pickle', 'wb') 
    pickle.dump(scaler, filehandler)
    filehandler.close()
    print("Scaler object saved.")

model_auth = input("Do you want to save the model? (Y/N)")

if model_auth == "Y" or model_auth == "y":
    #Saving the CatBoost model object
    filehandler = open('CatBoostRegressor.pickle', 'wb') 
    pickle.dump(cb, filehandler)
    filehandler.close()
    print("Model object saved.")