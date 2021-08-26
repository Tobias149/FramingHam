import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from scipy.sparse.csr import csr_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import urllib.request

## Loading of the relevant features and the target feature
df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/framingham_cleaned.csv')

x=df.drop('TenYearCHD',axis=1)
y=df.TenYearCHD

## Looking at target variable "TenYearCHD"
print("Distribution of the TenYearCHD: ", Counter(y))

## Training and test data for ADASYN
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)

## Standardization of xtrain
scaler_model = StandardScaler().fit(xtrain)
standard_X = scaler_model.transform(xtrain)
standard_X_test = scaler_model.transform(xtest)

## define the model
model_RF = RandomForestClassifier(random_state=42)

## How to find the optimum hyperparameter of random forest via GridSearch
params = { 
    #'n_estimators': range(100,110), # Number fo trees in the forrest
    'n_estimators': [60,70,80], # Number fo trees in the forrest
    'max_features': ['auto', 'sqrt', 'log2'], # number of features to consider when looking for the best split
    'max_depth' : [4,5,6,7,8], # maximum depth of tree
    'criterion' :['gini', 'entropy'], # quality of split
    'class_weight' :[{0: 1, 1: 1},{0: 1, 1: 5},{0: 1, 1: 3},{0: 1, 1: 2}] # Weights associated with classes
}

CV_model = GridSearchCV(estimator=model_RF, param_grid=params, cv=5, scoring='f1')
CV_model.fit(standard_X, ytrain)

Opt_estimator = CV_model.best_params_['n_estimators']
Opt_maxfeat = CV_model.best_params_['max_features']
Opt_maxdepth = CV_model.best_params_['max_depth']
Opt_crit = CV_model.best_params_['criterion']
Opt_class_weight = CV_model.best_params_['class_weight']

print(CV_model.best_params_)

## define model with optimized hyperparameter
model_RF = RandomForestClassifier(n_estimators=Opt_estimator,max_features=Opt_maxfeat, 
                                max_depth=Opt_maxdepth, criterion=Opt_crit, 
                                class_weight=Opt_class_weight)

## fit the model
model_RF.fit(standard_X, ytrain)

## make a single prediction
newrow = standard_X_test
ypred = model_RF.predict(newrow)

## evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model_RF, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

## Accuracy
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
## crosstab
print(pd.crosstab(ytest, ypred))

## Save the model to file wokring directory
#pickle.dump(model_RF, open('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/model_RF.pkl', 'wb'))

## Load the Model back from file
#pickled_model_RF = pickle.load(open('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/model_RF.pkl', 'rb'))
urllib.request.urlretrieve("https://raw.githubusercontent.com/Tobias149/FramingHam/main/Data%20science%20models/model_RF.pkl", "model_RF.pkl")
pickled_model_RF = pickle.load(open('model_RF.pkl', 'rb'))

## Create a test array for prediction 
test_person = [1,23,2,1,5,1,1,1,1,190,140,90,23,80,70]
test_person_predict = np.array(test_person)

## Make a prediction
prediction = pickled_model_RF.predict(test_person_predict)
print(prediction)