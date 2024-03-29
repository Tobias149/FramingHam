import pandas as pd
import numpy as np
from matplotlib import pyplot
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

# Loading of the relevant features and the target feature
df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/framingham_cleaned.csv')
x = df.drop(columns=['TenYearCHD'], axis=1) # dataset without the target feature
y = df['TenYearCHD'] # target feature

# Looking at target variable "TenYearCHD"
print("Distribution of the TenYearCHD: ", Counter(y))

# Training and test data for ADASYN
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)

knn = neighbors.KNeighborsRegressor(n_neighbors=5) #to define the classifier object

# Standardization of xtrain
scaler_knn = StandardScaler().fit(xtrain)
standard_X = scaler_knn.transform(xtrain)

# How to find the optimum k (n_neighbors) for ADAYSN ! Using of GridSearchCV
knn_grid = GridSearchCV(estimator = knn,
                param_grid={'n_neighbors': np.arange(1,20)}, cv=5)
knn_grid.fit(standard_X,ytrain)
kOptimum = knn_grid.best_params_['n_neighbors']
print(knn_grid.best_params_)
print(-knn_grid.best_score_)
print(knn_grid.scorer_)

#Declaration of ADASYN (Oversampling) object and fitting on dataset
ada = ADASYN(sampling_strategy='minority',random_state=5, n_neighbors = kOptimum)
xres, yres = ada.fit_resample(xtrain,ytrain)

# Standardization of xtrain (xres) und xtest
scaler = StandardScaler().fit(xres)
standard_X = scaler.transform(xres)
standard_X_test = scaler.transform(xtest)

# Find the optimum k (n_neighbors) value 
clf_neigh = KNeighborsClassifier()
clf_neigh.fit(standard_X,yres)
#kOptimum = clf_neigh.get_params
#print(clf_neigh.get_params)

# Prediction
ypred = clf_neigh.predict(standard_X_test) #

#Distribution of synthetic generation Oversampling
counter = Counter(yres)
print("Oversampled distribution of the TenYearCHD: ", Counter(yres))

# Accuracy
print(accuracy_score(ytest, ypred))
#
print(pd.crosstab(ytest, ypred))
