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

## Round values of all float variables within the framingham dataset
df['education'] = round(df['education'])
df['cigsPerDay'] = round(df['cigsPerDay'])
df['BPMeds'] = round(df['BPMeds'])
df['totChol'] = round(df['totChol'])
df['sysBP'] = round(df['sysBP'])
df['diaBP'] = round(df['diaBP'])
df['BMI'] = round(df['BMI'])
df['heartRate'] = round(df['heartRate'])
df['glucose'] = round(df['glucose'])

## Splitting the dataset into test and train set and creating a test dataset as csv file
y=df.TenYearCHD
x=df.drop('TenYearCHD',axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)

## Merging of xtest and yest again to the test dataframe
df_test_dataset = pd.DataFrame(xtest) # transfer xtest into dataframe
df_ytest_dataset = pd.DataFrame(ytest) # transfer ytest into dataframe

df_test_dataset.insert (15,"TenYearCHD", df_ytest_dataset) # Merge xtest with ytest to the train dataset
print(df_test_dataset.head())
df_test_dataset.isnull().sum()
#df_test_dataset.info()

# Creating the test dataset csv.file
df_test_dataset.rename(columns={'male':'Sex'}, inplace = True)
df_test_dataset.to_csv(r'/Users/tobiasschmidt/Desktop/framingham_test.csv', index = False, header=True)

## Working forward with the train dataset
## Creating of train dataset 
df_train_dataset = pd.DataFrame(xtrain)
df_ytrain_dataset = pd.DataFrame(ytrain)
df_train_dataset.insert (15,"TenYearCHD", df_ytrain_dataset)

print(df_train_dataset.isnull().sum())# showing an overview of NaN values of the dataset

## Beginning of cleaning of the train dataset
## 1) Column "male": Renaming of column "male"
df_train_dataset.rename(columns={'male':'Sex'}, inplace = True)

## 2) Column "cigsPerDay": Replace NaN with the mean of all smokers
    # Calculation of mean
mean_without_notSmokers = df_train_dataset[df_train_dataset.cigsPerDay>0].cigsPerDay.mean()
    # And replace NaN values with this mean
df_train_dataset['cigsPerDay']= np.where((df_train_dataset.currentSmoker == 1) & (df_train_dataset.cigsPerDay.isnull()), mean_without_notSmokers, df_train_dataset.cigsPerDay)
df_train_dataset.isnull().sum()

## 3) Column "totChol": Handling of NaN values in this column
    # Calculation of mean
mean_totChol = df_train_dataset.totChol.mean()
    # And replace NaN values with this mean
df_train_dataset['totChol']= np.where((df_train_dataset.totChol.isnull()), mean_totChol, df_train_dataset.totChol)
df_train_dataset.isnull().sum()

## 4) Column "BMI": Handling of NaN values in this column

# BMI Tabelle (Einteilung nach NRC, Diet and Health. Implications for Reducing Chronic Disease Risk)
# https://ladr.de/service/rechenprogramme/bmi

i = 1
VAR_1 = 19
VAR_2 = 24
VAR_max = df_train_dataset['age'].max()

while i < 7:

  if VAR_1 == 19: # Start Age of 19
      mean_BMI = df_train_dataset[(df_train_dataset.age >= VAR_1) & (df_train_dataset.age <= VAR_2)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df_train_dataset['BMI']= np.where((df_train_dataset.age >= VAR_1) & (df_train_dataset.age <= VAR_2) & (df_train_dataset.BMI.isnull()), mean_BMI, df_train_dataset.BMI)
      
  elif VAR_1 >= 25 and VAR_1 <= 54:
      mean_BMI = df_train_dataset[(df.age >= VAR_1) & (df_train_dataset.age <= VAR_2)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df_train_dataset['BMI']= np.where((df_train_dataset.age >= VAR_1) & (df_train_dataset.age <= VAR_2) & (df_train_dataset.BMI.isnull()), mean_BMI, df_train_dataset.BMI)

  elif VAR_1 == 55:
      mean_BMI = df_train_dataset[(df_train_dataset.age >= VAR_1) & (df_train_dataset.age <= VAR_2)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df_train_dataset['BMI']= np.where((df_train_dataset.age >= VAR_1) & (df_train_dataset.age <= VAR_2) & (df_train_dataset.BMI.isnull()), mean_BMI, df_train_dataset.BMI)

  elif VAR_1 == 66:
      mean_BMI = df_train_dataset[(df_train_dataset.age >= 66)& (df_train_dataset.age <= VAR_max)].BMI.mean()
      mean_BMI = round(mean_BMI, 2)
      df_train_dataset['BMI']= np.where((df_train_dataset.age >= VAR_1) & (df_train_dataset.age <= VAR_max) & (df_train_dataset.BMI.isnull()), mean_BMI, df_train_dataset.BMI)

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


## 5) Column "Education": Handling of NaN values
    # Replace NaN values with "1"
df_train_dataset['education']= np.where((df_train_dataset.education.isnull()), 1, df_train_dataset.education)
df_train_dataset.isnull().sum()

## 6) Column "heartRate": Handling of NaN values in this column
    # Delete row
df_train_dataset = df_train_dataset.dropna(subset=['heartRate'], axis=0)

## 7) Column "BPMeds": Handling of NaN values in this column

VAR_BPMeds = 0
float(VAR_BPMeds)
df_train_dataset['BPMeds']= np.where((df_train_dataset.BPMeds.isnull()), VAR_BPMeds, df_train_dataset.BPMeds)
df_train_dataset.isnull().sum()

print("Nan Values without Glucose: ", df_train_dataset.isnull().sum())

## 8) Column "glucose": Handling of NaN values in this column

plt.hist(df_train_dataset['glucose'],100)
plt.show

sns.scatterplot(data=df_train_dataset, x="glucose", y="TenYearCHD")

sns.scatterplot(data=df_train_dataset, x="glucose", y="age")

    # KNN

df_train_dataset_g=df_train_dataset.dropna()
x=df_train_dataset_g.drop(columns=['glucose','TenYearCHD'], axis=1)
y=df_train_dataset_g['glucose']

# Training and test data g = glucose
xtrain_g, xtest_g, ytrain_g, ytest_g = train_test_split(x,y,test_size=0.3,random_state=1)

# Standardization of xtrain & xtest
scaler = StandardScaler().fit(xtrain_g)
standard_X = scaler.transform(xtrain_g)
standard_X_test = scaler.transform(xtest_g)

# Create the model
#knn = neighbors.KNeighborsClassifier(n_neighbors=5) # 
knn = neighbors.KNeighborsRegressor(n_neighbors=5) #to define the classifier object

# How to find the optimum k (n_neighbors) value! Using of GridSearchCV
knn_grid = GridSearchCV(estimator = knn, scoring='neg_mean_squared_error',
                param_grid={'n_neighbors': np.arange(1,20)}, cv=5)
knn_grid.fit(standard_X,ytrain_g)
kOptimum = knn_grid.best_params_['n_neighbors']
print(knn_grid.best_params_)
print(-knn_grid.best_score_)
print(knn_grid.scorer_)

#Create the model with optimum k (n_neighbors)
knn = neighbors.KNeighborsRegressor(n_neighbors = kOptimum)

# Model fitting
knn.fit(standard_X, ytrain_g) # to fit only on training data

#Prediction
ypred = knn.predict(standard_X_test)
print('Prediction: ', ypred)

#Evaluate Model Performance
print('MSE_KNN:',mean_squared_error(ytest_g, ypred)) #Regression Metric
#print('CVS: ', cross_val_score(knn, xtrain, ytrain, cv=5)) #Cross-Validation

Glucose_mean = ytrain_g.mean()

print('MSE_Glucose_original:',mean_squared_error(ytest_g, [Glucose_mean]*len(ytest_g)))
# Selected all features with NaN value in column "glucose"

print(df_train_dataset.isnull().sum())

df_train_dataset.loc[df_train_dataset['glucose'].isna(),:] 

# Creation of modified Dataset (only rows with NaN value of glucose) and saving this dataset without column 'glucose','TenYearCHD'
df2 = df_train_dataset.loc[df_train_dataset['glucose'].isna(),:]
dfx = df2.drop(columns=['glucose','TenYearCHD'], axis=1)

# Perform trained KNN of new created dataset dfx and predict values for NaN
dfx = scaler.transform(dfx)
glucose_pred = knn.predict(dfx)

# Replace all NaN values in the origin dataset with the predicted values from KNN
df_train_dataset.loc[df_train_dataset['glucose'].isna(),'glucose'] = glucose_pred

# Change float variables into integers values in the train dataset

df_train_dataset['education'] = df_train_dataset['education'].astype("int")
df_train_dataset['cigsPerDay'] = df_train_dataset['cigsPerDay'].astype("int")
df_train_dataset['BPMeds'] = df_train_dataset['BPMeds'].astype("int")
df_train_dataset['totChol'] = df_train_dataset['totChol'].astype("int")
df_train_dataset['sysBP'] = df_train_dataset['sysBP'].astype("int")
df_train_dataset['diaBP'] = df_train_dataset['diaBP'].astype("int")
df_train_dataset['BMI'] = df_train_dataset['BMI'].astype("int")
df_train_dataset['heartRate'] = df_train_dataset['heartRate'].astype("int")
df_train_dataset['glucose'] = df_train_dataset['glucose'].astype("int")


# Saving of the cleaned Dataset on your Desktop
df_train_dataset.to_csv(r'/Users/tobiasschmidt/Desktop/framingham_cleaned.csv', index = False, header=True)

print("End")