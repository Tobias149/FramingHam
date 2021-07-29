import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = None #with this line, all columns are showed

df=pd.read_csv('C:/Users/Lisa Mary/Documents/TechLabs/Framingham Data Set/Framingham_cleaned.csv')
y=df.TenYearCHD
x=df.drop(columns=['TenYearCHD',  'metabolic_syndrome', 'unhealthy_behavior'])
xtrain_c, xtest_c, ytrain_c, ytest_c = train_test_split(x, y, test_size=0.3, random_state=5)
xtrain_c = pd.DataFrame(xtrain_c)
xtrain_c = xtrain_c.sort_index()
print(xtrain_c.head())


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
neigh.fit(xtrain_c, ytrain_c)
ypred=neigh.predict(xtest_c)


from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(sampling_strategy=0.2)
x_under, y_under = under.fit_resample(xtrain_c, ytrain_c)

#https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy=0.5)
#mit Undersampling --> Undersampling bringt schlechtere Resultate
xtrain_smote1, ytrain_smote1=smote.fit_resample(x_under, y_under)

#ohne Undersampling
xtrain_smote, ytrain_smote=smote.fit_resample(xtrain_c, ytrain_c)

from collections import Counter
print('Before SMOTE:', Counter(ytrain_c))
print('After Undersampling and SMOTE:', Counter(ytrain_smote1))
print('After SMOTE:', Counter(ytrain_smote))

#ohne Undersampling und SMOTE
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest_c, ypred))
print(pd.crosstab(ytest_c, ypred))

#mit SMOTE
neigh.fit(xtrain_smote, ytrain_smote)
ypred=neigh.predict(xtest_c)
print("Accuracy SMOTE:", accuracy_score(ytest_c, ypred))
print("SMOTE:", pd.crosstab(ytest_c, ypred))

#mit SMOTE und Undersampling
neigh.fit(xtrain_smote1, ytrain_smote1)
ypred1=neigh.predict(xtest_c)
print("Accuracy SMOTE and Undersampling:", accuracy_score(ytest_c, ypred1))
print("SMOTE and Undersampling:", pd.crosstab(ytest_c, ypred1))

#SMOTE NC for categorical features
from imblearn.over_sampling import BorderlineSMOTE
bsmote = BorderlineSMOTE()
x_oversample_borderline, y_oversample_borderline = bsmote.fit_resample(xtrain_c, ytrain_c)
neigh.fit(x_oversample_borderline, y_oversample_borderline)
ypred2=neigh.predict(xtest_c)
print("Accuracy Score Borderline SMOTE:", accuracy_score(ytest_c, ypred2))
print("Borderline SMOTE:", pd.crosstab(ytest_c, ypred2))


from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE()
X_oversample_svm, y_oversample_svm = svmsmote.fit_resample(xtrain_c, ytrain_c)
from sklearn.linear_model import LogisticRegression
classifier_svm = LogisticRegression()
classifier_svm.fit(X_oversample_svm, y_oversample_svm)
ypred3=classifier_svm.predict(xtest_c)
print("Accuracy SVM SMOTE:", accuracy_score(ytest_c, ypred3))
print("SVM SMOTE:", pd.crosstab(ytest_c, ypred3))

#cluster based oversampling method based on Santos et al. (2015)
#https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29
#https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
#https://gist.github.com/michiexile/5635273

# gap.py
# (c) 2013 Mikael Vejdemo-Johansson
# BSD License
#
# SciPy function to compute the gap statistic for evaluating k-means clustering.
# Gap statistic defined in
# Tibshirani, Walther, Hastie:
#  Estimating the number of clusters in a data set via the gap statistic
#  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423
'''
from sklearn.cluster import KMeans, Birch
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=xtrain_c.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(xtrain_c)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal




k, gapdf = optimalK(x, nrefs=5, maxClusters=15)
print ('Optimal k is: ', k)

'''

plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()

