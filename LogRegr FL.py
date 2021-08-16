import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
pd.options.display.max_columns = None

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

#import cleaned dataset
df = pd.read_csv("https://raw.githubusercontent.com/Tobias149/FramingHam/main/framingham_cleaned.csv")
df = df.iloc[: , 1:]
df.head()

#plot distribution of Ten Year Risk of coronary heart disease
ax = df['TenYearCHD'].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
ax.set_title('Ten Year Risk of coronary heart disease (0=no, 1=yes)', size=20, pad=30)
ax.set_ylabel('Number of patients', fontsize=14)

for i in ax.patches:
    ax.text(i.get_x(), i.get_height(), str(round(i.get_height(), 2)), fontsize=15)

from sklearn.preprocessing import MinMaxScaler

#scale columns that have values greater than 1 to [0, 1] range
to_scale = [col for col in df.columns if df[col].max() > 1]
mms = MinMaxScaler()
scaled = mms.fit_transform(df[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)

#Replace original columns with scaled ones
for col in scaled:
df[col] = scaled[col]

#split data
from sklearn.model_selection import train_test_split

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f'''% Positive class in Train = {np.round(y_train.value_counts(normalize=True)[1] * 100, 2)}
% Positive class in Test  = {np.round(y_test.value_counts(normalize=True)[1] * 100, 2)}''')

#% Positive class in Train = 14.84
#% Positive class in Test  = 14.56

#Logistic regression WITHOUT SMOTE
#result is a Binomial probability between 0 and 1 for the example belonging to class=1.
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
preds = logisticRegr.predict(X_test)

# Evaluate
print(f'Accuracy = {accuracy_score(y_test, preds):.2f}\nRecall = {recall_score(y_test, preds):.2f}\n')
from sklearn.metrics import classification_report
print(classification_report(y_test, logisticRegr.predict(X_test)))

#Accuracy = 0.86
#Recall = 0.04
#model can correctly classify almost all people without TenYearCHD
#it also classified 86% of all people WITH risk of TenYearCDH false

#Logistic Regression WITH SMOTE
from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=42)
sm=SMOTE(sampling_strategy=0.5, random_state=2)
X_sm, y_sm = sm.fit_resample(X, y)

print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_sm.shape}''')

y_sm = pd.DataFrame(y_sm)
print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100

#Shape of X before SMOTE: (2965, 14)
#Shape of X after SMOTE: (3790, 14)

#Balance of positive and negative classes (%):
#0    66.675462
#1    33.324538
#dtype: float64

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_sm, y_sm, test_size=0.25, random_state=42)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train1, y_train1)
preds = logisticRegr.predict(X_test1)

print(f'Accuracy = {accuracy_score(y_test1, preds):.2f}\nRecall = {recall_score(y_test1, preds):.2f}\n')
print(classification_report(y_test1, logisticRegr.predict(X_test1)))

#Accuracy = 0.70
#Recall = 0.36

#model can correctly classify 70% of all people without TenYearCHD
#it classified 36% of all people WITH risk of TenYearCDH false

#optimize hyper parameters of a Logistic Regression model using Grid Search
#https://www.dezyre.com/recipes/optimize-hyper-parameters-of-logistic-regression-model-using-grid-search-in-python

#std_slc = StandardScaler()
# remove the outliners and scale the data

#pca = decomposition.PCA()
#Principal Component Analysis(PCA) which will reduce the dimension of features

#logistic_Reg = linear_model.LogisticRegression()
#Logistic Regression as a Machine Learning model to use GridSearchCV


#pipe = Pipeline(steps=[('std_slc', std_slc),
                           #('pca', pca),
                           #('logistic_Reg', logistic_Reg)])
#Pipeline helps by passing modules one by one through GridSearchCV for which we want to get the best parameters


# -->  define the parameters that we want to optimise for these three objects

#StandardScaler doesnt require any parameters to be optimised by GridSearchCV

#n_components = list(range(1,X.shape[1]+1,1))

#Principal Component Analysis requires a parameter 'n_components' to be optimised.
#'n_components' signifies the number of components to keep after reducing the dimension

#C = np.logspace(-4, 4, 50)
#penalty = ['l1', 'l2']

#Logistic Regression requires two parameters 'C' and 'penalty' to be optimised by GridSearchCV.
#So we have set these two parameters as a list of values form which GridSearchCV will select the best value of parameter

#parameters = dict(pca__n_components=n_components,
                      #ogistic_Reg__C=C,
                      #logistic_Reg__penalty=penalty)

#creating a dictionary to set all the parameters options for different modules

#clf_GS = GridSearchCV(pipe, parameters)
#clf_GS.fit(X, y)

#print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
#print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
#print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
#print(); print(clf_GS.best_estimator_.get_params()['logistic_Reg'])

#-> result:
#Best Penalty: l2
#Best C: 0.18420699693267145
#Best Number Of Components: 13

#LogisticRegression(C=0.18420699693267145, class_weight=None, dual=False,
                   #fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   #max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   #random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   #warm_start=False)

#Accuracy = 0.70
#Recall = 0.33
