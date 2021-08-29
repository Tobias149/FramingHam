import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from numpy import mean
from numpy import std
pd.options.display.max_columns = None

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

df=pd.read_csv("/content/drive/MyDrive/Techlabs/framingham_cleaned.csv")

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

#X_train, X_test, y_train, y_test = train_test_split(
    #X, y, test_size=0.25, random_state=11)

#print(f'''% Positive class in Train = {np.round(y_train.value_counts(normalize=True)[1] * 100, 2)}
#% Positive class in Test  = {np.round(y_test.value_counts(normalize=True)[1] * 100, 2)}''')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=11)
df_test=pd.read_csv("/content/drive/MyDrive/Techlabs/framingham_test.csv")
x_test=df_test.drop("TenYearCHD", axis=1)
y_test=df_test.TenYearCHD

from sklearn.preprocessing import StandardScaler

#Standardization
scaler_model = StandardScaler().fit(X)
standard_X = scaler_model.transform(X)
standard_X_test = scaler_model.transform(x_test)

#Define Logistic Regression as a Machine Learning model to use GridSearchCV

logistic_Reg = linear_model.LogisticRegression()

#std_slc = StandardScaler()
# remove the outliners and scale the data

pipe = Pipeline(steps=[#('std_slc', std_slc),
                           ('logistic_Reg', logistic_Reg)])
#Pipeline helps by passing modules one by one through GridSearchCV for which we want to get the best parameters


C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']

#Logistic Regression requires two parameters 'C' and 'penalty' to be optimised by GridSearchCV.
#So we have set these two parameters as a list of values form which GridSearchCV will select the best value of parameter

parameters = dict(logistic_Reg__C=C,
                    logistic_Reg__penalty=penalty)

#creating a dictionary to set all the parameters options for different modules

clf_GS = GridSearchCV(pipe, parameters, cv=5, scoring="f1")
clf_GS.fit(standard_X, y)

print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print(); print(clf_GS.best_estimator_.get_params()['logistic_Reg'])

# define model with optimized hyperparameter
logistic_Reg = LogisticRegression(C=1.2067926406393288, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

# fit the model
logistic_Reg=logistic_Reg.fit(standard_X, y)

# make a single prediction on the test data
newrow = x_test
ypred = logistic_Reg.predict(newrow)

print(ypred)


from sklearn.metrics import f1_score
f1_score = f1_score(y_test, ypred, average="micro")
f1_score

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, ypred)
cnf_matrix
