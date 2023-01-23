# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:29:02 2021

@author: sanji
"""

from pandas import read_csv # Read data from a csv
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
import pickle
import matplotlib.pyplot as plt

#User prompt---------------------------------------------------------------------------------------
#User is being guided to select appropriate flag(1/2) so that they can select proper files
print("Enter 1 for Prediction (Required: Normalized force-displacement file only)")
print("Enter 2 for Evaluation (Required: Normalized force-displacement and normalized traction-separation files)")
is_true = True

while is_true:
    flag = input("Enter a number (1 or 2): ")
    flag = int(flag)
    if flag == 1:
        X_raw_path = input("Enter your normalized force-displacement file name: ")#normalized_Force.csv
        is_true = False
    elif flag == 2:
        X_raw_path = input("Enter your normalized force-displacement file name: ")
        y_raw_path = input("Enter your normalized traction-separation file name:")#normalized_traction.csv
        is_true = False
    else:
        print("Please enter number 1 or 2")

#Data Read------------------------------------------------------------------------------------------
#reading the input csv files(force-discplecement and traction-separation)
X_raw = read_csv(X_raw_path, header =None)
if flag == 2:
    y_raw = read_csv(y_raw_path, header =None)

#Data conversion from pandas to numpy array----------------------------------------------------------
X = X_raw.values
print("Original forece-displacement shape: ",X.shape)
if flag == 2:
    y = y_raw.values
    print("Original traction-separation shape: ",y.shape)

#Processing normalized data---------------------------------------------------------------------------      
a_FEA = 25
h_FEA = 5
E_FEA = 2000    
row, col = X.shape
Force=np.zeros((row, col))
for i in range(row):
    for j in range(col):
        Force[i,j]= ((X[i,j]*E_FEA*h_FEA**3)/a_FEA**2)
d= input("Enter normalized displacement(last value in the range):")
Displacement= float(d)*a_FEA


#Dimension conversion---------------------------------------------------------------------------------
#If the number of features  of the user's force-displacement data is not equal to 1000 than bleow code will convert the dimension to 1000
FD=[]
x1 = np.linspace(0, Displacement, len(Force[0]))
for j in range (len(X)):
    fd_points = []
    x_points = np.linspace(0, 10, 1000)
    for i in x_points:
        if Displacement >= i and Displacement <= 10:
            c = np.interp(i, x1, list(Force[j]))
            fd_points.append(c)
        else:
            fd_points.append(0)
            
    FD.append(fd_points)
FD = np.array(FD)
print("Converted forece-displacement shape: ",FD.shape)

#If the number of features  of the user's traction-separation data is not equal to 500 than bleow code will convert the dimension to 500
if flag == 2:
    TS=[]
    sp= input("Enter normalized separation(last value in the range):")
    separation = float(sp)
    x2 = np.linspace(0, separation, len(y[0]))
    for j in range (len(y)):
        ts_points = []
        x_points1 = np.linspace(0, 0.024, 500)#Normalized separation (e.g., s= 0.6mm, 0.6/25)
        for i in x_points1:
            if separation >= i and separation <= 0.024:
                s = np.interp(i, x2, list(y[j]))
                ts_points.append(s)
            else:
                ts_points.append(0)
        TS.append(ts_points)
    TS = np.array(TS)
    print("Converted traction-separation shape: ",TS.shape)


#Trained model loading-----------------------------------------------------------------------------------
#below code will load the pre-trained model(multioutputregressor.pkl)
print('Model loading.....')
file_name = "multioutputregressor.pkl"
multioutputregressor = pickle.load(open(file_name, "rb"))


#Prediction on user data----------------------------------------------------------------------------------
print('Prediction is started....')
predictions = multioutputregressor.predict(FD)

# Normalized prediction
predictions= predictions/E_FEA
np.savetxt('normalized_prediction.csv', predictions, delimiter=",")
print('Prediction is completed and saved at normalized_prediction.csv (in current directory)')    
#Validation------------------------------------------------------------------------------------------------
if flag == 2:
    #error calculation by importing function
    print('\nError Calculation')
    r = r2_score(TS,predictions,multioutput='variance_weighted')
    print("R2_score: ",r)
    # RMSE Computation 
    rmse = np.sqrt(MSE(TS, predictions))  
    print("RMSE: {}".format(rmse))

#Visualize normalized predicted T-S laws (as well as, actual normalized T-S laws if available)-----------------------------------    
for i in range(len(predictions)):
    plt.figure(figsize=(5,5))
    x = np.linspace(0, 0.024, 501)#Normalized seperation (e.g., s= 0.6mm, 0.6/25)
    plt.plot(x,[0]+list(predictions[i]), ls=('dashed'), lw=2, label='Prediction-'+str(i))
    if flag == 2:
        plt.plot(x,[0]+list(TS[i]), label='Actual-'+str(i))
    plt.xlabel('Separation',fontsize=24)
    plt.ylabel('Traction',fontsize=24)
    plt.legend()
    plt.show()
    plt.close()
