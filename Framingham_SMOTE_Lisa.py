import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = None #with this line, all columns are showed

df=pd.read_csv('C:/Users/Lisa Mary/Documents/TechLabs/Framingham Data Set/Framingham_cleaned.csv')
y=df.TenYearCHD
x=df.drop(columns=['TenYearCHD'])
xtrain_c, xtest_c, ytrain_c, ytest_c = train_test_split(x, y, test_size=0.3, random_state=5)
xtrain_c = pd.DataFrame(xtrain_c)
xtrain_c = xtrain_c.sort_index()
ytrain_c=pd.Series(ytrain_c).sort_index()
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


########################################################################################################################
########################################################################################################################
#cluster based oversampling method based on Santos et al. (2015)
#https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29
#https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
#https://gist.github.com/michiexile/5635273

#GAP statistic for finding the optimal value of k for k means

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
def optimalK(xtrain_c, nrefs=3, maxClusters=30):
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




k, gapdf = optimalK(xtrain_c, nrefs=3, maxClusters=10)
print ('Optimal k is: ', k)



plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()
'''


#GAP Method to find the optimal value of k for KMeans
#https://github.com/milesgranger/gap_statistic/blob/master/Example.ipynb
xtrain_c=xtrain_c.astype(float)
from gap_statistic import OptimalK
optimalK = OptimalK(n_jobs=4, parallel_backend='joblib')
n_clusters = optimalK(xtrain_c, n_refs=3, cluster_array=np.arange(1, 30))
print('Optimal clusters: ', n_clusters)
# 10 Wiederholungen Ergebnisse: 5,6,2,7,7,6,2,8,6,6

print(optimalK.gap_df.head())

plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
            optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
#plt.show()


########################################################################################################################
########################################################################################################################
#Elbow method, Silhouette method, Davies Bouldin Score
#https://becominghuman.ai/3-minute-read-to-how-to-find-optimal-number-of-clusters-using-k-means-algorithm-eaa6bdce92cc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
X=xtrain_c
wss = []
db = []
sil = []
K = range(2, 10)
for n in K:
    algorithm = (KMeans(n_clusters = n) )
    algorithm.fit(X)
    labels = algorithm.labels_
    db.append(davies_bouldin_score(X,labels))
    sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    wss.append(algorithm.inertia_)


#Visualization

fig, (ax1, ax2, ax3) = plt.subplots(ncols =3)
# fig, (ax3) = plt.subplots(ncols =1)
fig.set_figheight(10)
fig.set_figwidth(30)

ax1.plot(K, wss, 'bo')
ax1.plot(K, wss, 'r-', alpha = 0.5)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Sum_of_squared_distances')
ax1.set_title('Elbow Method For Optimal k')
ax1.grid(True)

ax2.plot(K,sil, 'bo')
ax2.plot(K, sil, 'r-', alpha = 0.5)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Method For Optimal k')
ax2.grid(True)

ax3.plot(K,db, 'bo')
ax3.plot(K, db, 'r-', alpha = 0.5)
ax3.set_xlabel('Number of Clusters (k)')
ax3.set_ylabel('DB index')
ax3.set_title('DB Index Method For Optimal k')
ax3.grid(True)

#plt.show()

#k optimal aus den Abbildungen(5 Wiederholungen): 5,3,3,3,3


#########################################################################################################################
########################################################################################################################
#Clustering with KMeans
#https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=123)
X=xtrain_c
pred_y = kmeans.fit_predict(X)
print(pred_y)
print(np.unique(pred_y, return_counts=True))

#Dividing the clusters for further analysis
cluster_1=(xtrain_c[pred_y==0])
cluster_1y=(ytrain_c[pred_y==0])
cluster_2=(xtrain_c[pred_y==1])
cluster_2y=(ytrain_c[pred_y==1])
cluster_3=(xtrain_c[pred_y==2])
cluster_3y=(ytrain_c[pred_y==2])
cluster_4=(xtrain_c[pred_y==3])
cluster_4y=(ytrain_c[pred_y==3])
cluster_5=(xtrain_c[pred_y==4])
cluster_5y=(ytrain_c[pred_y==4])
cluster_6=(xtrain_c[pred_y==5])
cluster_6y=(ytrain_c[pred_y==5])

#Joining the clusters with the target variable
cluster_1=cluster_1.join(cluster_1y)
cluster_2=cluster_2.join(cluster_2y)
cluster_3=cluster_3.join(cluster_3y)
cluster_4=cluster_4.join(cluster_4y)
cluster_5=cluster_5.join(cluster_5y)
cluster_6=cluster_6.join(cluster_6y)

print(cluster_1.shape)  #512 lines
print(cluster_1['TenYearCHD'].sum()) #70
print(cluster_2.shape)  #640 lines
print(cluster_2['TenYearCHD'].sum()) #76
print(cluster_3.shape)  #484 lines
print(cluster_3['TenYearCHD'].sum()) #46
print(cluster_4.shape)  #157 lines
print(cluster_4['TenYearCHD'].sum()) #26
print(cluster_5.shape)  #21 lines
print(cluster_5['TenYearCHD'].sum()) #14
print(cluster_6.shape)  #262 lines
print(cluster_6['TenYearCHD'].sum()) #79

#erster Durchlauf: cluster_1: 512(20), cluster_2: 640(76), cluster_3:484(46), cluster_4:157(26), cluster_5:(21(14), cluster_6:262(79)

########################################################################################################################
########################################################################################################################
#Oversampling of the newly created clusters

smote=SMOTE(sampling_strategy=('all'), k_neighbors=3)
X=cluster_2
y=cluster_2y
xtrain_cbs, ytrain_cbs=smote.fit_resample(X, y)
print(xtrain_cbs['TenYearCHD'].isnull().sum)
