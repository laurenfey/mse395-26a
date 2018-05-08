'''
@author Lauren Smith
Uses input files and training data to test different sci-kit learn models

MSE 395 -- Group 26a
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso

from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


#function that does that model fitting
def fit(data,regr):
    data = np.nan_to_num(data)
    np.random.shuffle(data)
    MP_ID = data[:,1]
    target = data[:,-1]
    descriptors = data[:,2:-1]

    #remove testing data
    fitting_descriptors,testing_descriptors,fitting_target,testing_target = train_test_split(descriptors,target,test_size=0.1)

    chunk_size = int(len(fitting_target)/9)
    idx = np.arange(len(fitting_target))

    validation_err = np.zeros(9)
    training_err = np.zeros(9)
    testing_err = np.zeros(len(testing_target))

    for i in range(9):
        #split data into validation and training sets
        validation_target = fitting_target[i*chunk_size:(i+1)*chunk_size]
        validation_descriptors = fitting_descriptors[i*chunk_size:(i+1)*chunk_size]
        training_target = fitting_target[np.logical_or(idx < i*chunk_size,idx >= (i+1)*chunk_size)]
        training_descriptors = fitting_descriptors[np.logical_or(idx < i*chunk_size,idx >= (i+1)*chunk_size)]

        regr.fit(training_descriptors,training_target)

        validation_predict = regr.predict(validation_descriptors)
        training_predict = regr.predict(training_descriptors)
        testing_predict = regr.predict(testing_descriptors)

        validation_err[i] = np.mean(np.abs(validation_predict-validation_target))
        training_err[i] = np.mean(np.abs(training_predict-training_target))
        #testing_err[i] = np.mean(np.abs(testing_predict-testing_target))
        testing_err += np.abs(testing_predict-testing_target)

    testing_err /= 9

    regr.fit(training_descriptors,training_target)

    testing_predict = regr.predict(testing_descriptors)
    validation_predict = regr.predict(validation_descriptors)
    training_predict = regr.predict(training_descriptors)

    plt.figure(figsize=(6,6))
    plt.xlabel('Actual Optical Phonon Frequency (1/cm)',fontsize=12)
    plt.ylabel('Predicted Optical Phonon Frequency (1/cm)',fontsize=12)
    plt.plot([0,2000],[0,2000])
    plt.errorbar(training_target,training_predict,label='Training',fmt='o')
    plt.errorbar(validation_target,validation_predict,fmt='o',label='Validation')
    plt.errorbar(testing_target,testing_predict,yerr=testing_err,fmt='o',label="Testing")
    plt.legend(loc = 2,fontsize=14)
    print('Testing data average % Error: '+str(np.mean(testing_err/testing_target)*100))
    print('Testing data MAE: '+str(np.mean(testing_err)))
    print('Validation data MAE: '+str(np.mean(validation_err)))
    print('Training data MAE: '+str(np.mean(training_err)))
    plt.show()




data = np.genfromtxt('TrainingData.csv',delimiter=',')

#Test different models here
regr = DecisionTreeRegressor(max_depth=6)
print('Decision Tree')
fit(data,regr)
regr = RandomForestRegressor(max_depth=3, random_state=0)
print('\n\nRandom Forest')
fit(data,regr)
regr = MLPRegressor(solver='lbfgs', alpha=1e-2,hidden_layer_sizes=(2,8,8,2), random_state=1)
print('\n\nNeural Network')
fit(data,regr)
print('\n\nLasso')
regr = Lasso(alpha=10000)
fit(data,regr)
print("\n\nRidge")
fit(data,regr)
regr = linear_model.LinearRegression()
print("\n\nLinear Regression")
fit(data,regr)

#calculate error bars here
n = 1000
total = []
for i in range(n):
    total.append(fit(data,regr))
print(np.sum(total)/n,np.std(total)/np.sqrt(len(total))) #average training error, standard error on training error
