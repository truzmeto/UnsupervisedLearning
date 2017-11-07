import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_samples, silhouette_score
import time
import sys

# pass path+filenames as argument for processing
train_path = sys.argv[1]
test_path =  sys.argv[2]

# load data
train = pd.read_csv(train_path ,index_col=False)
test = pd.read_csv(test_path,index_col=False)


train_labels = train['income'].values
train = train.drop('income',axis=1).values
test_labels = test['income'].values
test = test.drop('income',axis=1).values

#parameters
n_centroids = 2
n_seed = 100

#normalize everything
a = train
train = (a - a.mean()) / np.std(a)
b = test
test = (b - b.mean()) / np.std(b)


#apply k-means
kmeans = KMeans(n_clusters=n_centroids, n_init=1, n_jobs=4, random_state=n_seed)
kmeans.fit(train)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))

pred_test = kmeans.predict(test)
print(confusion_matrix(test_labels, pred_test))
print(classification_report(test_labels, pred_test))


# choosing number of clusters via Silhoutte + Model Complexity
nclusters = 50
nfeatures = 65
data_frac = 5000
X = train 
X = X[0:data_frac]
y = train_labels

sil = []
time_clust = []
for iclusters in range(2,nclusters):
    start = time.time()
    clusterer = KMeans(n_clusters=iclusters,random_state=n_seed, n_init=1)
    cluster_labels = clusterer.fit_predict(X)
    end = time.time()
    sil.append(silhouette_score(X, cluster_labels))
    time_clust.append(end - start) 
    


# model complexity
time_iter = []
time_sample = []
niter = nclusters # just for plotting them togather
for iiter in range(2,niter):
    start = time.time() # iterations
    clusterer = KMeans(n_clusters= 2, random_state=iiter, n_init = iiter, max_iter = 100)
    cluster_labels = clusterer.fit_predict(X)
    end = time.time()
    time_iter.append(end - start)    
    start = time.time() # samples
    isample = int(data_frac * iiter/100)
    clusterer = KMeans(n_clusters= 2, random_state=n_seed, n_init = 1, max_iter = 100)
    cluster_labels = clusterer.fit_predict(X[0:isample])
    end = time.time()
    time_sample.append(end - start)      
       

# visualize silhouette + model complexity
y = np.asarray(sil)
x = range(2,nclusters)
sil_max = x[y.argmax()]

sns.set_style("whitegrid")
fig = plt.figure(figsize=(11,4))
p1 = plt.subplot(121, title = 'Silhoutte')
plt.plot(x, y)
plt.ylabel('Silhouette score(width)')
plt.xlabel('Number of clusters')
plt.axvline(x=sil_max,color='magenta', linestyle='--', linewidth=0.5)

p2 = plt.subplot(122, title = 'K-means O(knT)')
plt.plot(time_clust,x)
plt.plot(time_iter,x)
plt.plot(time_sample,x)
plt.legend(["N-clusters","N-iterations","Data Size %"],loc=4)
plt.xlabel('CPU time in sec ')
plt.ylabel('y')

fig.set_tight_layout(True)
plt.show()
fig.savefig('plots/ModComp_Kmeans.pdf')
