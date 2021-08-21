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

# Loading of the relevant features and the target feature
df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/framingham_cleaned.csv')

x=df.drop('TenYearCHD',axis=1)
y=df.TenYearCHD

# Looking at target variable "TenYearCHD"
print("Distribution of the TenYearCHD: ", Counter(y))

# Training and test data for ADASYN
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)

# define the model
model = RandomForestClassifier(n_estimators=10,criterion="entropy")

# Standardization of xtrain
scaler_model = StandardScaler().fit(xtrain)
standard_X = scaler_model.transform(xtrain)
standard_X_test = scaler_model.transform(xtest)

# fit the model
model.fit(standard_X, ytrain)

# make a single prediction
newrow = standard_X_test
ypred = model.predict(newrow)

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Accuracy
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# crosstab
print(pd.crosstab(ytest, ypred))