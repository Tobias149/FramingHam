import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skn
from sklearn.model_selection import train_test_split
df = pd.read_csv('framingham.csv')
y=df.TenYearCHD
x=df.drop('TenYearCHD',axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=5)
print(xtrain.head())
#print(data)
#data.info()
#data['education'].value_counts()
#print(len(data.drop_duplicates())) # Anzahl der Duplicate Anzeigen
#data.iloc[-1] # Funktioniert nicht!
#print(data[data['age']> 50]) # Funkioniert nicht!!
#print(data.describe()) # Funktioniert nicht!!
#print(data["age"].dtypes)# Funktioniert nicht!!
#data.dropna(subset=["education"], axis=0)
#data.info()
#data.to_csv(r'/Users/philinehoefling/Desktop/Dataset/New_framingham.csv', index = False, header=True)