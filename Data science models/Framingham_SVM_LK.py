import pandas as pd
from sklearn.model_selection import train_test_split

#Datset with SMOTE
df = pd.read_csv('C:/Users/Lisa Mary/Documents/TechLabs/Framingham Data Set/Framingham_SMOTE.csv')
df.columns=["index","Sex","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes",
            "totChol","sysBP","diaBP","BMI","heartRate","glucose","TenYearCHD"]
df=df.drop(columns=['index'])

#Dataset without SMOTE
#df = pd.read_csv('C:/Users/Lisa Mary/Documents/TechLabs/Framingham Data Set/framingham_cleaned.csv')
y = df.TenYearCHD
x = df.drop(columns=['TenYearCHD'])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=5)
xtrain = pd.DataFrame(xtrain)
xtrain = xtrain.sort_index()
ytrain = pd.Series(ytrain).sort_index()
print(xtrain.head())

# Standardization of xtrain
from sklearn.preprocessing import StandardScaler
scaler_model = StandardScaler().fit(xtrain)
stand_xtrain= scaler_model.transform(xtrain)
stand_xtest = scaler_model.transform(xtest)

# define the model
from sklearn import svm
model = svm.SVC()

'''
# hyperparameter optimization GridSearch
#https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
#c tells the SVM optimisation how much error is bearable
#when gamma is higher, nearby points will have high influence on the calculation of the separation line
#The main function of the kernel is to take low dimensional input space and transform it into a higher-dimensional space
gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1')
gs.fit(stand_xtrain, ytrain)

print(gs.best_params_)
#für Dataset SMOTE {'C': 0.1, 'gamma': 0.1, 'kernel': 'poly
#für Dataset ohne SMOTE {'C': 1, 'gamma': 0.01, 'kernel': 'poly'}
'''

# define model with optimized hyperparameter
#model = svm.SVC(C=0.1, kernel='poly', gamma=0.1) #with SMOTE
model = svm.SVC(C=1, kernel='poly', gamma=0.01) #without SMOTE

# fit the model
model.fit(stand_xtrain, ytrain)

# make a single prediction
ypred = model.predict(stand_xtest)

# evaluate the model
from sklearn.metrics import f1_score
f1_score=f1_score(ytest, ypred, average='micro')
print(f1_score)
#results SMOTE: 0.85        without SMOTE:0.85

from sklearn.metrics import roc_auc_score
roc_auc_score=roc_auc_score(ytest, ypred)
print(roc_auc_score)
#results SMOTE: 0.50        without SMOTE:0.51

from sklearn.metrics import recall_score
recall=recall_score(ytest, ypred)
print(recall_score)

# crosstab
print(pd.crosstab(ytest, ypred))

#without smote
'''
0.015037593984962405
0.5006604679002281
<function recall_score at 0x000000EDF79D60D0>
col_0         0  1
TenYearCHD        
0           758  5
1           126  1
'''

#mit cb_smote
'''
0.8215434083601286
0.8106287061339554
<function recall_score at 0x000000BA8ED46040>
col_0         0    1
TenYearCHD          
0           423   85
1           137  511
'''