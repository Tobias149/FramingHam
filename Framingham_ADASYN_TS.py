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

# Loading of the relevant features and the target feature
df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/Cleaned_framingham.csv')
x = df.drop(columns=['TenYearCHD'], axis=1) # dataset without the target feature
y = df['TenYearCHD'] # target feature

# Looking at target variable "TenYearCHD"
print("Distribution of the TenYearCHD: ", Counter(y))

# Training and test data for ADASYN
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)

# Standardization of xtrain
scaler = StandardScaler().fit(xtrain)
standard_X = scaler.transform(xtrain)

# Find the optimum k (n_neighbors) value 
clf_neigh = KNeighborsClassifier()
clf_neigh.fit(standard_X,ytrain)
kOptimum = clf_neigh.get_params
print(clf_neigh.get_params)

# Prediction
ypred = clf_neigh.predict(xtest)

#Declaration of ADASYN (Oversampling) object and fitting on dataset
ada = ADASYN(sampling_strategy='minority',random_state=5, n_neighbors = 5)
xres, yres = ada.fit_resample(xtrain,ytrain)

#Distribution of synthetic generation Oversampling
counter = Counter(yres)
print("Oversampled distribution of the TenYearCHD: ", Counter(yres))

# Accuracy
print(accuracy_score(ytest, ypred))