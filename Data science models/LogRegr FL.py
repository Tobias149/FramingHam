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


df = pd.read_csv("/content/drive/MyDrive/Techlabs/framingham_test.csv")
df = df.iloc[: , 1:]
df.head

ax = df['TenYearCHD'].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
ax.set_title('Ten Year Risk of coronary heart disease (0=no, 1=yes)', size=20, pad=30)
ax.set_ylabel('Number of patients', fontsize=14)

for i in ax.patches:
    ax.text(i.get_x(), i.get_height(), str(round(i.get_height(), 2)), fontsize=15)

from sklearn.model_selection import train_test_split

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=11)

print(f'''% Positive class in Train = {np.round(y_train.value_counts(normalize=True)[1] * 100, 2)}
% Positive class in Test  = {np.round(y_test.value_counts(normalize=True)[1] * 100, 2)}''')

from sklearn.preprocessing import StandardScaler

#Standardization of X_Train
scaler_model = StandardScaler().fit(X_train)
standard_X = scaler_model.transform(X_train)
standard_X_test = scaler_model.transform(X_test)

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

clf_GS = GridSearchCV(pipe, parameters, cv=5, scoring='recall')
clf_GS.fit(X, y)

print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print(); print(clf_GS.best_estimator_.get_params()['logistic_Reg'])

# define model with optimized hyperparameter
LogisticRegression(C=75.43120063354607, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

# fit the model
logistic_Reg.fit(X, y)


# make a single prediction
newrow = X
ypred = logistic_Reg.predict(newrow)

print(ypred)

# evaluate the model
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=11)
n_scores = cross_val_score(logistic_Reg, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')


# Accuracy
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# crosstab
print(pd.crosstab(y, ypred))