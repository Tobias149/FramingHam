
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = None #with this line, all columns are showed

df=pd.read_csv('C:/Users/Lisa Mary/Documents/TechLabs/Framingham Data Set/framingham.csv')
y=df.TenYearCHD
x=df.drop('TenYearCHD', axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=5)
xtrain = pd.DataFrame(xtrain)
xtrain = xtrain.sort_index()
print(df.head())
print(df.dtypes)
print(df.describe())

#Missing values
#Heatmap null values
#cols=xtrain.columns
#missingvalues=plt.figure(2)
#sns.heatmap(xtrain[cols].isnull()).set_title('missing values')
print(xtrain.isnull().sum())

#Features
#SEX
print(xtrain['male'].value_counts())
#fig_sex=plt.figure(1)
#sns.countplot(x='male', data=xtrain).set_title('sex')


#AGE
print(xtrain['age'].describe())
#fig_age=plt.figure(3)
#sns.countplot(y="age", data=xtrain).set_title('age')


#EDUCATION
print(xtrain['education'].value_counts())
#fig_education=plt.figure(4)
#sns.countplot(x='education', data=xtrain).set_title('education')

#Missing values
print(xtrain['education'].isnull().value_counts())
missedu=xtrain[xtrain['education'].isnull()]
#print(missedu.head(30)) #no correlation between missing features in single individuals


#SMOKER
print(xtrain['currentSmoker'].value_counts())
smoker=xtrain[xtrain['currentSmoker']==1]
#fig_smoker=plt.figure(5)
#sns.countplot(x="currentSmoker", data=xtrain).set_title('currentSmoker')


#CIGSPERDAY
smoker=smoker['cigsPerDay']
print(smoker.describe())
print(smoker)

nonsmoker=xtrain[xtrain['cigsPerDay']==0]
nonsmoker=nonsmoker['cigsPerDay']
print(nonsmoker)

#fig_cigspersmoker=plt.figure(12)
#ax = sns.boxplot(x="currentSmoker", y="cigsPerDay", data=xtrain)


#BPMEDS
print(xtrain['BPMeds'].value_counts())
#fig_BPmeds=plt.figure(6)
#sns.countplot(x="BPMeds", data=xtrain).set_title('BPMeds')


#PREVALENTSTROKE
print(xtrain['prevalentStroke'].value_counts())
#fig_prevalentStroke=plt.figure(7)
#sns.countplot(x='prevalentStroke', data=xtrain).set_title('prevalentStroke')


#PREVALENTHYP
print(xtrain['prevalentHyp'].value_counts())
#fig_prevalentHyp=plt.figure(8)
#sns.countplot(x='prevalentHyp', data=xtrain).set_title('prevalentHyp')

#Correlation prevalent Hypertension and BP Medication
#Count of individuals with Hypertension
Hypcount=xtrain[xtrain['prevalentHyp']==1].value_counts(['prevalentHyp'])
print(Hypcount)
Hyp=xtrain[xtrain['prevalentHyp']==1]
#Count of hypertensive individuals with medication
HypMed=Hyp[Hyp['BPMeds']==1].value_counts(['BPMeds'])
print(HypMed)
#Share of hypertensive individuals with BP medication
PercMed=HypMed/Hypcount
print(PercMed)
PercNoMed=1-PercMed

#Pie Chart hypertensive individuals with medication
piedata=([0.090426, 1-0.090426])
#fig_medication=plt.figure(9)
#mylabels = ["BPMeds", "No medication"]
#plt.pie(piedata, labels=mylabels, autopct='%1.1f%%').set_title('BPMeds')


#DIABETES
print(xtrain['diabetes'].value_counts())
#fig_diabetes=plt.figure(10)
#sns.countplot(x='diabetes', data=xtrain).set_title('diabetes')

#Correlation diabetes and glucose levels --> high glucose levels=diagnosed diabetes
highgluc=xtrain[xtrain['glucose']>200]
print(highgluc[['glucose', 'diabetes']])
print(highgluc.value_counts([highgluc['glucose']>200]))
print(highgluc.value_counts('diabetes'))

#TOTCHOL
print(xtrain['totChol'].describe())
#fig_totchol=plt.figure(13)
#sns.boxplot(y="totChol", data=xtrain).set_title('TotChol')

#fig_corr_chol_BMI=plt.figure(14)
#plt.scatter(x='BMI', y='totChol', data=xtrain)


#SYSBP
print(xtrain['sysBP'].describe())
#fig_sysBP=plt.figure(14)
#sns.boxplot(x='prevalentHyp', y="sysBP", data=xtrain).set_title('prevalentHyp')


#DIABP
print(xtrain['diaBP'].describe())
#fig_diaBP=plt.figure(15)
#sns.boxplot(x='prevalentHyp', y="diaBP", data=xtrain).set_title('diaBP')


#BMI
print(xtrain['BMI'].describe())
#fig_BMI=plt.figure(16)
#sns.boxplot(y="BMI", data=xtrain)

#heartRate
print(xtrain['heartRate'].describe())
#fig_heartrate=plt.figure(17)
#sns.boxplot(y="heartRate", data=xtrain)

#glucose
print(xtrain['glucose'].describe())
#fig_glucose=plt.figure(18)
#sns.boxplot(x='diabetes', y="glucose", data=xtrain)



#Fill missing values
#education
xtrain['education']=xtrain['education'].fillna(xtrain['education'].mode()[0])
print(xtrain['education'].isnull().sum())

#cigsPerDay
print((xtrain['cigsPerDay'].isnull().sum()))
print(nonsmoker.isnull().sum())
print(smoker.isnull().sum())
#--> alle 21 fehlenden Werte im xtrain Datensatz sind von Rauchern, bei Nichtrauchern gibt es keine fehlenden Werte
cigsperday_smoker=int(smoker.mean())
print(cigsperday_smoker) #18 Zigaretten pro Tag im Durchschnitt bei Rauchern
xtrain['cigsPerDay']=xtrain['cigsPerDay'].fillna(cigsperday_smoker)
print(xtrain['cigsPerDay'].isnull().sum())

#BPMeds
xtrain['BPMeds']=xtrain['BPMeds'].fillna(0)
print(xtrain['BPMeds'].isnull().sum())

#TotChol
xtrain['totChol']=xtrain['totChol'].fillna(xtrain['totChol'].mean())
print(xtrain['totChol'].isnull().sum())



#BMI
print(xtrain['age'].describe()) #Age 32-70 1:32-44/2:45-57/3:58-70
#define age groups
bins= [30,44,57,71]
labels = ['age1','age2','age3']
xtrain['AgeGroup'] = pd.cut(xtrain['age'], bins=bins, labels=labels, right=False)
print (xtrain)

BMI_age1=xtrain[xtrain['AgeGroup']=='age1'].BMI.median()
BMI_age2=xtrain[xtrain['AgeGroup']=='age2'].BMI.median()
BMI_age3=xtrain[xtrain['AgeGroup']=='age3'].BMI.median()

print(BMI_age1)
print(BMI_age2)
print(BMI_age3)

xtrain[xtrain['AgeGroup'] == 'age1']['BMI'].fillna(BMI_age1)
xtrain[xtrain['AgeGroup'] == 'age2']['BMI'].fillna(BMI_age2)
xtrain[xtrain['AgeGroup'] == 'age3']['BMI'].fillna(BMI_age3)


if 'AgeGroup' == 'age1':
    xtrain['BMI'].fillna(BMI_age1, inplace=True)
elif 'AgeGroup' == 'age2':
    xtrain['BMI'].fillna(BMI_age2, inplace=True)
else:
    xtrain['BMI'].fillna(BMI_age3, inplace=True)

print(xtrain[xtrain['BMI'].isnull()])
print(xtrain['BMI'].isnull().sum())

#Heartrate
print(xtrain['heartRate'].index[xtrain['heartRate'].apply(np.isnan)].tolist())
xtrain=xtrain.dropna(subset=['heartRate'])
print(xtrain.describe())
print(xtrain['heartRate'].isnull().sum())


#Glucose
xtrain=xtrain.drop('AgeGroup', axis=1)
test_df=xtrain[xtrain['glucose'].isnull()]
data=xtrain.dropna()

y_train=data['glucose']
x_train=data.drop('glucose', axis=1)
x_test=test_df.drop('glucose', axis=1)


#knn für missing
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor()
xtrain_no_nan=xtrain.dropna()
Y=xtrain_no_nan.glucose
X=xtrain_no_nan.drop(['glucose'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
x_train = pd.DataFrame(x_train)
y_train= pd.DataFrame(y_train)
x_test = pd.DataFrame(x_test)
y_test= pd.DataFrame(y_test)
print(x_train, x_test, y_train, y_test)

#Datenstandardisierung
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#y_train=scaler.fit_transform(y_train)
#y_test=scaler.transform(y_test)


'''
#GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
mse=mean_squared_error
k_range = list(range(1, 31))
parameters= {
    'n_neighbors': k_range,
    'weights':['uniform', 'distance'],
    'metric':['euclidean', 'manhattan']
}

grid=GridSearchCV(KNeighborsRegressor(), parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)


#fitting the model for gridsearch
grid_results=grid.fit(x_train, y_train)
print(grid_results.best_params_)
'''
#n_neighbors: 18, metric: euclidean, weights: distance



#knn - optimal number of neighbors
knn = KNeighborsRegressor(n_neighbors=18, weights='distance', metric='euclidean')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
#y_pred=scaler.inverse_transform(y_pred)
#y_test=scaler.inverse_transform(y_test)
y_pred=pd.DataFrame(y_pred)
print(y_pred.head())

#Model evaluation
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))

#Evaluation of mean
mean_glucose=[xtrain['glucose'].mean() for number in range(0, 807)]
print(mean_squared_error(y_test, mean_glucose))

#Train knn on whole xtrain and define rows with missing glucose values as x_test
print(xtrain.isnull().sum())
y_train1=xtrain_no_nan.glucose
x_train1=xtrain_no_nan.drop(['glucose'], axis=1)

x_test1=xtrain[xtrain['glucose'].isnull()].drop(['glucose'], axis=1)
print(x_test1)
print(x_train1, y_train1)

#Standardization
x_train1=scaler.fit_transform(x_train1)
x_test1=scaler.transform(x_test1)

#knn
knn = KNeighborsRegressor(n_neighbors=18, weights='distance', metric='euclidean')
knn.fit(x_train1, y_train1)
y_pred1 = knn.predict(x_test1)
y_pred1=pd.DataFrame(y_pred1)


y_pred1['glucose']=y_pred1
y_pred1=y_pred1.drop(0, axis=1)
print(y_pred1)

#Fehlende Werte mit y_pred1 ersetzen
#xtrain['glucose'].fillna(y_pred1['glucose'], inplace=True)
#print(xtrain.isnull().head())
#https://stackoverflow.com/questions/14016247/find-integer-index-of-rows-with-nan-in-pandas-dataframe
index_NaN = xtrain['glucose'].index[xtrain['glucose'].apply(np.isnan)].tolist()
print(index_NaN)
#https://stackoverflow.com/questions/13842088/set-value-for-particular-cell-in-pandas-dataframe-using-index
for index, value in enumerate(y_pred1['glucose']):
    xtrain.at[index_NaN[index], 'glucose'] = value

print(xtrain['glucose'])


#Visualisierungen
'''
Heatmap null values
cols=xtrain.columns
missingvalues=plt.figure(2)
sns.heatmap(xtrain[cols].isnull()).set_title('missing values')
print(xtrain.isnull().sum())

#Verteilung der Geschlechter
fig_sex=plt.figure(1)
sns.countplot(x='male', data=xtrain).set_title('sex')

#Altersverteilung
#fig_age=plt.figure(3)
#sns.countplot(y="age", data=xtrain).set_title('age')

#Verteilung education
fig_education=plt.figure(4)
sns.countplot(x='education', data=xtrain).set_title('education')

#Verteilung Raucher/Nichtraucher
fig_smoker=plt.figure(5)
sns.countplot(x="currentSmoker", data=xtrain).set_title('currentSmoker')

#Zigaretten pro Tag
fig_cigspersmoker=plt.figure(12)
ax = sns.boxplot(x="currentSmoker", y="cigsPerDay", data=xtrain)

#BPMeds
fig_BPmeds=plt.figure(6)
sns.countplot(x="BPMeds", data=xtrain).set_title('BPMeds')

#prevalent Stroke
fig_prevalentStroke=plt.figure(7)
sns.countplot(x='prevalentStroke', data=xtrain).set_title('prevalentStroke')

#Hypertension
fig_prevalentHyp=plt.figure(8)
sns.countplot(x='prevalentHyp', data=xtrain).set_title('prevalentHyp')

#Pie Chart hypertensive individuals with medication
piedata=([0.090426, 1-0.090426])
fig_medication=plt.figure(9)
mylabels = ["BPMeds", "No medication"]
plt.pie(piedata, labels=mylabels, autopct='%1.1f%%')

#Diabetes
fig_diabetes=plt.figure(10)
sns.countplot(x='diabetes', data=xtrain).set_title('diabetes')

#Totchol
fig_totchol=plt.figure(13)
sns.boxplot(y="totChol", data=xtrain).set_title('TotChol')

#Correlation BMI and Cholesterin
fig_corr_chol_BMI=plt.figure(14)
plt.scatter(x='BMI', y='totChol', data=xtrain)

#SYSBP
fig_sysBP=plt.figure(14)
sns.boxplot(x='prevalentHyp', y="sysBP", data=xtrain).set_title('prevalentHyp')

#DIABP
fig_diaBP=plt.figure(15)
sns.boxplot(x='prevalentHyp', y="diaBP", data=xtrain).set_title('diaBP')

#BMI
fig_BMI=plt.figure(16)
sns.boxplot(y="BMI", data=xtrain)

#heartRate
fig_heartrate=plt.figure(17)
sns.boxplot(y="heartRate", data=xtrain)

#glucose
fig_glucose=plt.figure(18)
sns.boxplot(x='diabetes', y="glucose", data=xtrain)




#Verteilung der numerischen Werte

#age
fig_age_distribution=sns.displot(data=xtrain, x=xtrain['age'], kde=True).set(title='Age Distribution')
fig_age_distribution.fig.subplots_adjust(top=.9)
fig_age_distribution.set_axis_labels("Age", "Count")

#cigsPerDay
cigsPerDay_Smoker=xtrain.loc[xtrain["currentSmoker"] == 1, 'cigsPerDay']
#https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/03_subset_data.html
fig_cigsPerDay=sns.displot(data=xtrain, x=cigsPerDay_Smoker, kde=True).set(title='Cigarettes per Day')
fig_cigsPerDay.fig.subplots_adjust(top=.9)
fig_cigsPerDay.set_axis_labels("Cigarettes per Day", "Count")


#total cholesterol levels
fig_totchol=sns.displot(data=xtrain, x=xtrain['totChol'], kde=True).set(title='Distribution of cholesterol levels')
fig_totchol.fig.subplots_adjust(top=.9)
fig_totchol.set_axis_labels("Cholesterol level", "Count")

#systolic blood pressure
fig_sysbp=sns.displot(data=xtrain, x=xtrain['sysBP'], kde=True).set(title='Distribution of systolic blood pressure')
fig_sysbp.fig.subplots_adjust(top=.9)
fig_sysbp.set_axis_labels("Systolic blood pressure", "Count")

#diastolic blood pressure
fig_diabp=sns.displot(data=xtrain, x=xtrain['diaBP'], kde=True).set(title='Distribution of diastolic blood pressure')
fig_diabp.fig.subplots_adjust(top=.9)
fig_diabp.set_axis_labels("Diastolic blood pressure", "Count")

#BMI
fig_bmi=sns.displot(data=xtrain, x=xtrain['BMI'], kde=True).set(title='Distribution of BMI')
fig_bmi.fig.subplots_adjust(top=.9)
fig_bmi.set_axis_labels("Body mass index", "Count")

#Heart rate
fig_hr=sns.displot(data=xtrain, x=xtrain['heartRate'], kde=True).set(title='Distribution of the heart rate')
fig_hr.fig.subplots_adjust(top=.9)
fig_hr.set_axis_labels("Heart rate", "Count")

#Glucose
fig_gluc=sns.displot(data=xtrain, x=xtrain['glucose'], kde=True).set(title='Distribution of glucose levels')
fig_gluc.fig.subplots_adjust(top=.9)
fig_gluc.set_axis_labels("Glucose level", "Count")
'''



#Korrelationsmatrix
#neue Variable - metabolic_syndrome als mögliche Kombination von Risikofaktoren (nur 13 Teilnehmer identifiziert)
xtrain_1=xtrain
#xtrain_1['metabolic_syndrome'] = np.empty((len(xtrain_1), 0)).tolist()
xtrain_1['metabolic_syndrome'] = np.where((xtrain_1.BMI >30) & (xtrain_1.sysBP>140) & (xtrain_1.totChol>200) & (xtrain.diabetes==1), 1, 0)
print(xtrain_1.metabolic_syndrome)
print(xtrain_1['metabolic_syndrome'].sum())
print(xtrain_1.isnull().sum())

xtrain_1['unhealthy_behavior'] = np.where((xtrain_1.BMI >30) & (xtrain_1.currentSmoker==1) & (xtrain_1.totChol>200) & (xtrain.diabetes==1), 1, 0)
print(xtrain_1['unhealthy_behavior'].sum())

ytrain=ytrain.drop(index=689)
#xtrain_1=xtrain_1.drop('TenYearCHD', axis=1)
print(xtrain_1)

xtrain_1['TenYearCHD']=ytrain
print(xtrain_1)
corr = xtrain_1.corr()
print(corr)
'''
#Erstellung Korrelationsmatrix
sns.heatmap(corr, annot=True)


#Erstellung aller Figures
plt.show()
plt.close('all')


#Diverse Visualisierungen und Analysen
from pandas_profiling import ProfileReport
profile=ProfileReport(xtrain, title='Profile report heart disease')
profile.to_file('Profile report heart disease.html')
'''

#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Correlation with output variable
cor_target = abs(corr['TenYearCHD'])#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
print(relevant_features)


#'https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = xtrain_1  #independent columns
y = ytrain   #target column i.e price range#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


#Feature importance
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

xtrain.to_csv(r'C:\Users\Lisa Mary\Documents\TechLabs\Framingham Data Set\Framingham_cleaned.csv', index = False)