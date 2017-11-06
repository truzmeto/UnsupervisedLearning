
# coding: utf-8

# ## Loan Data

# In[69]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report
from mpl_toolkits.mplot3d import Axes3D # 3d scatter plot
import time
get_ipython().magic('matplotlib inline')


# In[104]:

# parameters
n_centroids = 2
n_init = 40
n_seed = 199


# In[105]:

train = pd.read_csv('./clean_data/loan_train.txt',index_col=False)
test = pd.read_csv('./clean_data/loan_test.txt',index_col=False)


# In[66]:

train_labels = train['loan_status'].values
train = train.drop('loan_status',axis=1).values
test_labels = test['loan_status'].values
test = test.drop('loan_status',axis=1).values


# In[67]:

# normalize everything such that categoricals are not affected
a = train
train = (a - a.mean()) / np.std(a)
b = test
test = (b - b.mean()) / np.std(b)


# In[ ]:




# ### Apply K Means

# In[5]:

kmeans = KMeans(n_clusters=n_centroids, n_init=1, random_state=n_seed, n_jobs=4)
kmeans.fit(train)


# In[6]:

print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# ### Predicting on test set

# In[7]:

b = kmeans.predict(test)


# In[8]:

print(confusion_matrix(test_labels, b))
print(classification_report(test_labels, b))


# ## Choosing number of clusters via Silhoutte + Model Complexity

# In[9]:

from sklearn.metrics import silhouette_samples, silhouette_score
nclusters = 50
nfeatures = 65
data_frac = 5000


# In[10]:

X = train #.values # converting df to np.array
X = X[0:data_frac]
y = train_labels


# In[11]:

sil = []
time_clust = []
for iclusters in range(2,nclusters):
    start = time.time()
    clusterer = KMeans(n_clusters=iclusters,random_state=n_seed,n_init=1)
    cluster_labels = clusterer.fit_predict(X)
    end = time.time()
    sil.append(silhouette_score(X, cluster_labels))
    time_clust.append(end - start) 
    


# In[12]:

time_iter = []
time_sample = []
niter = nclusters # just for plotting them togather
for iiter in range(2,niter):
    # iterations
    start = time.time()
    clusterer = KMeans(n_clusters= 2, random_state=iiter, n_init = iiter, max_iter = 100)
    cluster_labels = clusterer.fit_predict(X)
    end = time.time()
    time_iter.append(end - start)    
    #samples
    start = time.time()
    isample = int(data_frac * iiter/100)
    clusterer = KMeans(n_clusters= 2, random_state=n_seed, n_init = 1, max_iter = 100)
    cluster_labels = clusterer.fit_predict(X[0:isample])
    end = time.time()
    time_sample.append(end - start)      
       


# In[13]:

y = np.asarray(sil)
x = range(2,nclusters)
sil_max = x[y.argmax()]


# In[14]:

sns.set_style("whitegrid")
fig = plt.figure(figsize=(11,4))
p1 = plt.subplot(121, title = 'Loan Data Silhoutte')
plt.plot(x, y)
plt.ylabel('Silhouette score(width)')
plt.xlabel('Number of clusters')
plt.axvline(x=sil_max,color='magenta', linestyle='--', linewidth=0.5)

p2 = plt.subplot(122, title = 'Loan Data K-means O(knT)')
plt.plot(time_clust,x)
plt.plot(time_iter,x)
plt.plot(time_sample,x)
plt.legend(["N-clusters","N-iterations","Data Size %"],loc=4)
plt.xlabel('CPU time in sec ')
plt.ylabel('y')

fig.set_tight_layout(True)
plt.show()
fig.savefig('plots/LC_ModComp_Kmeans.pdf')


# **The average complexity is given by O(k n T), were n is the number of samples and T is the number of iteration.**

# ### Apply EM

# In[15]:

from sklearn.mixture import GaussianMixture


# In[16]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, n_init=1, max_iter=100)
gm.fit(train)


# In[17]:

gm_labels = gm.predict(train)


# In[18]:

print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# In[19]:

test_gm = gm.predict(test)
print(confusion_matrix(test_labels, test_gm))
print(classification_report(test_labels, test_gm))


# ## EM model Complexity + Choosing n-components

# In[20]:

BIC = []
AIC = []
time_components = []
ncomponents = 50
for icomponents in range(1,ncomponents):
    start = time.time()
    clusterer = GaussianMixture(n_components=icomponents, n_init = 1, covariance_type='full',random_state=n_seed).fit(X) 
    end = time.time()
    BIC.append(clusterer.bic(X))
    AIC.append(clusterer.aic(X))
    time_components.append(end - start) 


# In[21]:

time_iter = []
time_sample = []
niter = ncomponents  
for iiter in range(1,niter):
    # iterations
    start = time.time()
    clusterer = GaussianMixture(n_components = 2, n_init=iiter, covariance_type='full',random_state=iiter).fit(X) 
    end = time.time()
    time_iter.append(end - start)    
    #samples
    start = time.time()
    isample = int(data_frac * iiter/100)
    clusterer = GaussianMixture(n_components = 2, n_init = 1, covariance_type='full', random_state=n_seed).fit(X[0:isample]) 
    end = time.time()
    time_sample.append(end - start)      


# In[22]:

x = np.arange(1, ncomponents)
sns.set_style("whitegrid")
fig = plt.figure(figsize=(11,4))

p1 = plt.subplot(121, title = 'LC EM Gaussian Mixture choosing n_components')
plt.plot(x, BIC, label='BIC')
plt.plot(x, AIC, label='AIC')
plt.axvline(x=2,color='magenta', linewidth=0.5)
plt.legend(loc='best')
plt.xlabel('n_components')
plt.ylabel('y')

p2 = plt.subplot(122, title = 'LC EM Gaussian Mixture Model Complexity')
plt.plot(time_components,x, linestyle='--')
plt.plot(time_iter,x)
plt.plot(time_sample,x)
plt.legend(["N-components","N-iterations","Data Size %"],loc=4)
plt.xlabel('CPU time in sec ')
plt.ylabel('y')

fig.set_tight_layout(True)
plt.show()
fig.savefig('plots/LC_ModComp_EM.pdf')


# In[ ]:




# # Note!
# **Since training was not provided any information about target feature, it makes sence to see very close accuracy
# on both training and testing sets.**

# ### Apply PCA to Normalized Data 

# In[52]:

from sklearn.decomposition import PCA, FastICA


# In[24]:

train = pd.DataFrame(train)
n_features = len(train.columns)


# ## Try it on whole data

# In[25]:

# when n_comp is given as fraction and svd solver is full, then algorithm chooses # pc components
# such that model explains "n_components= of varience
pca_all = PCA(n_components=0.99,  svd_solver = 'full')
pca_all.fit(train)
pca_n = pca_all.transform(train)


# In[26]:

sns.set_style("whitegrid")
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pca_n[:,2], pca_n[:,0],pca_n[:,1],c=train_labels,cmap='plasma')
ax1.set_xlabel('PC2')
ax1.set_ylabel('PC0')
ax1.set_zlabel('PC1')
plt.title('LC Best Principal Components')

ax2 = fig.add_subplot(122)
aig_vals = pca_all.explained_variance_ratio_
y_pos = np.arange(len(aig_vals))
cumsum = np.cumsum(aig_vals)  
plt.bar(y_pos, aig_vals, align='edge', alpha=0.8, color='red', width=-0.3)
plt.bar(y_pos, cumsum, align='edge', alpha=0.8, width=0.3)
plt.legend(["Fraction of explained variance","Cumulative Sum for Explained variance fraction"],loc='best')
plt.ylabel('')
plt.ylim(0, 1.2 )
plt.xlabel('Principal Components') 
plt.title('LC Moddel Explained Var. by PCs')
fig.set_tight_layout(True)
fig.savefig('plots/LC_pca.pdf')


# ### Apply k-means to PCA output

# In[27]:

kmeans = KMeans(n_clusters=n_centroids,n_init=1, random_state=n_seed)
kmeans.fit(pca_n)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# ### Apply EM to PCA output

# In[28]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, n_init=1, max_iter=100)
gm.fit(pca_n)
gm_labels = gm.predict(pca_n)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# ##  ICA

# In[29]:

ica =FastICA(algorithm='parallel', tol=0.001, whiten=True, fun='logcosh', max_iter=50, random_state=n_seed)
ica_all = ica.fit_transform(train)


# ### Applying k-means to ICA output

# In[30]:

kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(ica_all)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# ### Applying EM to ICA output

# In[31]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, max_iter=100)
gm.fit(ica_all)
gm_labels = gm.predict(ica_all)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# In[32]:

from scipy.stats import kurtosis
y = kurtosis(ica_all, fisher=True)
x = np.arange(len(y)) 


# In[33]:

kurt_max = x[y.argmax()]
kurtosis_thresh = y.max()/2


# ### Visualize ICA components with highest Kurtosis

# In[34]:

tmp = [y > kurtosis_thresh]
indx = np.where(tmp)[1]


# In[35]:

ica_all = pd.DataFrame(ica_all)
ica_keep = ica_all[indx]


# In[36]:

indx


# In[37]:

fig = plt.figure(figsize=(13,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(ica_keep[indx[0]], ica_keep[indx[1]],ica_keep[indx[2]],c=train_labels,cmap='plasma')
ax1.set_xlabel('IC0')
ax1.set_ylabel('IC1')
ax1.set_zlabel('IC2')
plt.title('LC 3 Best Independent Components')

ax2 = fig.add_subplot(122)
plt.xlabel('Independent components', fontsize = 14)
plt.ylabel('Kurtosis', fontsize = 14)
plt.plot(x ,y, '-',lw=2., color='blue')
plt.axvline(x=kurt_max,color='magenta', linestyle='--', linewidth=0.5)
plt.axhline(y=kurtosis_thresh,color='magenta', linestyle='--', linewidth=0.5)
plt.title('LC Kurtosis plot')
fig.set_tight_layout(True)
fig.savefig('plots/LC_ica.pdf')


# ### Apply k-means to reduced dim by ICA

# In[38]:

kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(ica_keep)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# ### Apply EM to reduced dim by ICA

# In[39]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, max_iter=100)
gm.fit(ica_keep)
gm_labels = gm.predict(ica_keep)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# In[40]:

ica_n = ica_keep


# In[ ]:




# ## Random Projections

# In[41]:

from sklearn import random_projection
import scipy.sparse as sps
from scipy.linalg import pinv
from collections import defaultdict


# In[42]:

def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)


# In[43]:

tmp = defaultdict(dict)
dims = train.shape[1]
for iseed in range(1,11):
    for dim in range(2,dims,2):
        rp = random_projection.GaussianRandomProjection(random_state=iseed, n_components=dim)
        rp.fit(train)  
        tmp[dim][iseed] = reconstructionError(rp, train)
tmp =pd.DataFrame(tmp).T


# In[44]:

fig = plt.figure(figsize=(5,4))
y = tmp.mean(axis=1)
x = np.arange(2,dims,2)
ax2 = fig.add_subplot(111)
plt.xlabel('Number of RP Components', fontsize = 14)
plt.ylabel('Reconstruction Error', fontsize = 14)
plt.plot(x ,y, '-o',lw=2., color='magenta')
plt.title('LC Reconstruction Error')
fig.set_tight_layout(True)
fig.savefig('plots/LC_rpa.pdf')


# In[45]:

rp_n = random_projection.GaussianRandomProjection(n_components=n_features,
                                                  eps=0.9,
                                                  random_state=n_seed).fit_transform(train)


# In[51]:

train.shape


# In[112]:

kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(rp_n)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# In[113]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, max_iter=100)
gm.fit(rp_n)
gm_labels = gm.predict(rp_n)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# ## Factor Analysis
# 

# In[56]:

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score


# In[47]:

fa = FactorAnalysis(random_state=n_seed)


# In[48]:

fa_scores = []
X=train[0:5000] # try it on smaller data subset
for n in np.arange(1,30):
    fa.n_components = n
    fa_scores.append(np.mean(cross_val_score(fa, X)))


# In[49]:

sns.set_style("whitegrid")
x, y = np.arange(0,len(fa_scores)), np.array(fa_scores)
score_max = x[y.argmax()]

fig = plt.figure(figsize=(5,4))
plt.plot(x, y)
plt.xlabel('Number of FA Components', fontsize = 14)
plt.ylabel('cv score', fontsize = 14)
plt.title('LC Factor analysis CV score')
plt.axvline(x=score_max,color='magenta', linestyle='--', linewidth=0.5)
fig.set_tight_layout(True)
fig.savefig('plots/LC_fa.pdf')


# In[118]:

fa_n = FactorAnalysis(n_components= score_max, random_state=n_seed).fit_transform(train)


# In[121]:

kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(fa_n)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# In[122]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, max_iter=100)
gm.fit(fa_n)
gm_labels = gm.predict(fa_n)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# ## Benchmark DR Algorithms

# In[57]:

time_comp_pca = []
time_sample_pca = []
time_comp_ica = []
time_sample_ica = []
time_comp_rp = []
time_sample_rp = []
time_comp_fa = []
time_sample_fa = []
ncomponents = 28 
data_frac = 5000
X = train #.values 
X = X[0:data_frac]

for icomponents in range(2,ncomponents):
    start = time.time() #components
    PCA(n_components=icomponents).fit_transform(X)
    end = time.time()
    time_comp_pca.append(end - start)
    start = time.time() # samples
    isample = int(data_frac * icomponents/100)
    PCA(n_components=3).fit_transform(X[0:isample])
    end = time.time()
    time_sample_pca.append(end - start)
    
    start = time.time() #components
    FastICA(n_components=icomponents,tol=0.001).fit_transform(X)
    end = time.time()
    time_comp_ica.append(end - start)
    start = time.time() # samples
    FastICA(n_components=3,tol=0.001).fit_transform(X[0:isample])
    end = time.time()
    time_sample_ica.append(end - start)
    
    start = time.time() #components
    random_projection.GaussianRandomProjection(n_components=icomponents).fit_transform(X)
    end = time.time()
    time_comp_rp.append(end - start)
    start = time.time() # samples
    random_projection.GaussianRandomProjection(n_components=3).fit_transform(X[0:isample]) 
    end = time.time()
    time_sample_rp.append(end - start)
    
    start = time.time() #components
    FactorAnalysis(n_components=icomponents).fit_transform(X)
    end = time.time()
    time_comp_fa.append(end - start)
    start = time.time() # samples
    FactorAnalysis(n_components=3).fit_transform(X[0:isample])
    end = time.time()
    time_sample_fa.append(end - start)
    
    
    
    


# In[62]:

x = np.arange(2, ncomponents)
sns.set_style("whitegrid")
fig = plt.figure(figsize=(6,4))

plt.subplot(111, title = 'LC Dim. Reduction Time Benchmarking')
plt.plot(time_comp_pca,x, linestyle='--',color='red')
plt.plot(time_sample_pca,x, linestyle='-',color='red')

plt.plot(time_comp_ica,x, linestyle='--',color='blue')
plt.plot(time_sample_ica,x, linestyle='-',color='blue')

plt.plot(time_comp_rp,x, linestyle='--',color='cyan')
plt.plot(time_sample_rp,x, linestyle='-',color='cyan')

plt.plot(time_comp_fa,x, linestyle='--',color='black')
plt.plot(time_sample_fa,x, linestyle='-',color='black')

plt.legend(["PCA Components","PCA data Size %",
            "ICA Components","ICA data Size %",
            "RP Components","RP data Size %",
            "FA Components","FA data Size %"],loc='best')
plt.xlabel('CPU time in sec ')
plt.ylabel('y')

fig.set_tight_layout(True)
plt.show()
fig.savefig('plots/LC_DR_benchmarks.pdf')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ## Apply ANN on Projected Data 

# In[ ]:



