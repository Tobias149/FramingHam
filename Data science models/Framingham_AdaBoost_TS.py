import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# Loading of the relevant features and the target feature
df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/framingham_cleaned.csv')

x=df.drop('TenYearCHD',axis=1)
y=df.TenYearCHD

# Looking at target variable "TenYearCHD"
print("Distribution of the TenYearCHD: ", Counter(y))

# Training and test data for ADASYN
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)

# Standardization of xtrain
scaler_model = StandardScaler().fit(xtrain)
standard_X = scaler_model.transform(xtrain)
standard_X_test = scaler_model.transform(xtest)

#model_Adaboost =  AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
model_Adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# How to find the optimum hyperparameter of AdaBoost via GridSearch
# The most important parameters are base_estimator, n_estimators, and learning_rate
params = { 
    #'base_estimator__max_depth':[i for i in range(2,11,2)],
    #'base_estimator__min_samples_leaf':[5,10],
    'n_estimators':[10,50,250],
    'learning_rate':[0.01,0.1,1]
}

CV_model = GridSearchCV(estimator=model_Adaboost, param_grid=params, cv=5, scoring='f1')
CV_model.fit(standard_X, ytrain)

#Opt_Base_estimator_depth = CV_model.best_params_['base_estimator__max_depth']
#Opt_Base_estimator_samp = CV_model.best_params_['base_estimator__min_samples_leaf']
Opt_estimators = CV_model.best_params_['n_estimators']
Opt_learning_rate = CV_model.best_params_['learning_rate']

print(CV_model.best_params_)

# define model with optimized hyperparameter
model_Adaboost = AdaBoostClassifier(n_estimators=Opt_estimators, 
                                    learning_rate=Opt_learning_rate)

# fit the model
model_Adaboost.fit(standard_X, ytrain)

# make a single prediction
ypred = model_Adaboost.predict(standard_X_test)

## evaluate the model
print("Accuracy:",metrics.accuracy_score(ytest, ypred))
print(pd.crosstab(ytest, ypred))