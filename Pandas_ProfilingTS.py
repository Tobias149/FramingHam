import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

df = pd.read_csv('/Users/tobiasschmidt/Desktop/TechLabs 2/Dataset 2/framingham_cleaned_280721.csv')

profile = ProfileReport(df, title="Framingham Profiling Report")

profile.to_file("Framingham_Report.html")