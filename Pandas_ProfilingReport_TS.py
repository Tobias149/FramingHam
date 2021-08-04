import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# Load the dataset
df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/Cleaned_framingham.csv')

# Load data into the Report
profile = ProfileReport(df, title="Framingham Profiling Report")

# Creation of the file: profile.to_file("your_report.html")
profile.to_file("Profiling_Report.html")

print("The report was created successfully! Look into your files")