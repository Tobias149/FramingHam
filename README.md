# FramingHam - Predicting a ten year risk of coronary heart disease
Which files are important in this repo regarding the framingham dataset? Look at the following folders...

#1. Folder Data cleaning:
#1.1 framingham_origin.csv (This is the original dataset downloaded from Kaggle (https://www.kaggle.com/dileep070/heart-disease-prediction-using-logistic-regression?select=framingham.csv; downloaded 29.05.2021 &nbsp;
#1.2 Framingham_Data cleaning_TS.py (Python program to clean the original dataset and split it into test (30%) and train (70%) data
#1.3 framingham_cleaned.csv (70% of the original dataset, also called train dataset)
#1.4 framingham_test.csv (30% of the original dataset, also called test dataset)
#1.5 Profiling_Report.html (Statistical analyzation of the original dataset) 

#2. Data science models:
#2.1 Framingham_RandomForest_TS.py (Python program regarding Random Forest Classifier)
#2.2 RandomForest_Model.pkl(Serializing of the trained machine learning model - Random Forest Classifier)

#3. Graphical user interface:
#3.1 Framingham_GUI_TS.py (Python program of the graphical user interface, which is connected with py.file Framingham_RandomForest_TS.py)
