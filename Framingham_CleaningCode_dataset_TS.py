import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
import seaborn as sns
import sklearn as skn
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/framingham.csv')

print(df.isnull().sum())# showing an overview of NaN values of the dataset

# 1) Column "male": Renaming of column
df.rename(columns={'male':'Sex'}, inplace = True)

# 2) Column "cigsPerDay": Replace NaN with the mean of all smokers
    # Calculation of mean
mean_without_notSmokers = df[df.cigsPerDay>0].cigsPerDay.mean()
    # And replace NaN values with this mean
df['cigsPerDay']= np.where((df.currentSmoker == 1) & (df.cigsPerDay.isnull()), mean_without_notSmokers, df.cigsPerDay)
df.isnull().sum()

# 3) Column "totChol": Handling of NaN values in this column
    # Calculation of mean
mean_totChol = df.totChol.mean()
    # And replace NaN values with this mean
df['totChol']= np.where((df.totChol.isnull()), mean_totChol, df.totChol)
df.isnull().sum()

# 4)  Column "BMI": Handling of NaN values in this column

x = df['age']
y = df['BMI']

plt.bar(x,y, color='red')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show

plt.hist(df['BMI'],100)
plt.show

# 5) Column "BMI": Handling of NaN values in this column

# BMI Tabelle (Einteilung nach NRC, Diet and Health. Implications for Reducing Chronic Disease Risk)
# https://ladr.de/service/rechenprogramme/bmi

i = 1
VAR_1 = 19
VAR_2 = 24
VAR_max = df['age'].max()

while i < 7:

  if VAR_1 == 19: # Start Age of 19
      mean_BMI = df[(df.age >= VAR_1) & (df.age <= VAR_2)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df['BMI']= np.where((df.age >= VAR_1) & (df.age <= VAR_2) & (df.BMI.isnull()), mean_BMI, df.BMI)
      
  elif VAR_1 >= 25 and VAR_1 <= 54:
      mean_BMI = df[(df.age >= VAR_1) & (df.age <= VAR_2)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df['BMI']= np.where((df.age >= VAR_1) & (df.age <= VAR_2) & (df.BMI.isnull()), mean_BMI, df.BMI)

  elif VAR_1 == 55:
      mean_BMI = df[(df.age >= VAR_1) & (df.age <= VAR_2)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df['BMI']= np.where((df.age >= VAR_1) & (df.age <= VAR_2) & (df.BMI.isnull()), mean_BMI, df.BMI)

  elif VAR_1 == 66:
      mean_BMI = df[(df.age >= 66)& (df.age <= VAR_max)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df['BMI']= np.where((df.age >= VAR_1) & (df.age <= VAR_max) & (df.BMI.isnull()), mean_BMI, df.BMI)

  print('Mean of BMI(NRC): ',mean_BMI,'\tAge from: ', VAR_1,' to ', VAR_2)

  if i == 1:
     VAR_1 = 25
     VAR_2 = 34
  elif i >= 2 and i <= 3:
     VAR_1 = VAR_1 + 10
     VAR_2 = VAR_2 + 10
  elif i == 4:
     VAR_1 = 55
     VAR_2 = 65
  elif i == 5:
     VAR_1 = 66
     VAR_2 = VAR_max 

  i += 1

print(df.isnull().sum())

# 6) Column "Education": Handling of NaN values
    # Replace NaN values with "1"
df['education']= np.where((df.education.isnull()), 1, df.education)
df.isnull().sum()

# 7) Column "heartRate": Handling of NaN values in this column
    # Delete row
df = df.dropna(subset=['heartRate'], axis=0)

# 8) Column "glucose": Handling of NaN values in this column

plt.hist(df['glucose'],100)
plt.show

sns.scatterplot(data=df, x="glucose", y="TenYearCHD")

sns.scatterplot(data=df, x="glucose", y="age")

    # KNN

df1=df.dropna()
x=df1.drop(columns=['glucose','TenYearCHD'], axis=1)
y=df1['glucose']

# Training and test data
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)

# Standardization of xtrain & xtest
scaler = StandardScaler().fit(xtrain)
standard_X = scaler.transform(xtrain)
standard_X_test = scaler.transform(xtest)

# Create the model
#knn = neighbors.KNeighborsClassifier(n_neighbors=5) # 
knn = neighbors.KNeighborsRegressor(n_neighbors=5) #to define the classifier object

# How to find the optimum k (n_neighbors) value! Using of GridSearchCV
knn_grid = GridSearchCV(estimator = knn, scoring='neg_mean_squared_error',
                param_grid={'n_neighbors': np.arange(1,20)}, cv=5)
knn_grid.fit(standard_X,ytrain)
kOptimum = knn_grid.best_params_['n_neighbors']
print(knn_grid.best_params_)
print(-knn_grid.best_score_)
print(knn_grid.scorer_)

#Create the model with optimum k (n_neighbors)
knn = neighbors.KNeighborsRegressor(n_neighbors = kOptimum)

# Model fitting
knn.fit(standard_X, ytrain) # to fit only on training data

#Prediction
ypred = knn.predict(standard_X_test)
print('Prediction: ', ypred)

#Evaluate Model Performance
print('MSE_KNN:',mean_squared_error(ytest, ypred)) #Regression Metric
#print('CVS: ', cross_val_score(knn, xtrain, ytrain, cv=5)) #Cross-Validation

Glucose_mean = ytrain.mean()

print('MSE_Glucose_original:',mean_squared_error(ytest, [Glucose_mean]*len(ytest)))
    # Selected all features with NaN value in column "glucose"
df.loc[df['glucose'].isna(),:] 

    # Creation of modified Dataset (only rows wiht NaN value of glucose) and saving this dataset without column 'glucose','TenYearCHD'
df2 = df.loc[df['glucose'].isna(),:]
dfx = df2.drop(columns=['glucose','TenYearCHD'], axis=1)

    # Perform trained KNN of new created dataset dfx and predict values for NaN
dfx = scaler.transform(dfx)
glucose_pred = knn.predict(dfx)

    # Replace all NaN values in the origin dataset with the predicted values from KNN
df.loc[df['glucose'].isna(),'glucose'] = glucose_pred

    # Saving of the cleaned Dataset on your Desktop
df.to_csv(r'/Users/tobiasschmidt/Desktop/Cleaned_framingham.csv', index = False, header=True)