
# coding: utf-8

# ## Adult Data

# In[85]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report
from mpl_toolkits.mplot3d import Axes3D # 3d scatter plot
import time
#get_ipython().magic('matplotlib inline')


# In[86]:

# parameters
n_centroids = 2
n_init = 40
n_seed = 100


# In[87]:

train = pd.read_csv('./clean_data/adult_train.txt',index_col=False)
test = pd.read_csv('./clean_data/adult_test.txt',index_col=False)


# In[88]:

# load training set
train_labels = train['income'].values
train = train.drop('income',axis=1).values

# load testing set
test_labels = test['income'].values
test = test.drop('income',axis=1).values


# In[89]:

# normalize everything such that categoricals are not affected
a = train
#train = (a - a.min()) / (a.max() - a.min())
train = (a - a.mean()) / np.std(a)

b = test
#test = (b - b.min()) / (b.max() - b.min())
test = (b - b.mean()) / np.std(b)


# ### Apply K Means

# In[5]:

kmeans = KMeans(n_clusters=n_centroids, n_init=1, n_jobs=4, random_state=n_seed)
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
    clusterer = KMeans(n_clusters=iclusters,random_state=n_seed, n_init=1)
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
p1 = plt.subplot(121, title = 'Adult Data Silhoutte')
plt.plot(x, y)
plt.ylabel('Silhouette score(width)')
plt.xlabel('Number of clusters')
plt.axvline(x=sil_max,color='magenta', linestyle='--', linewidth=0.5)

p2 = plt.subplot(122, title = 'Adult Data K-means O(knT)')
plt.plot(time_clust,x)
plt.plot(time_iter,x)
plt.plot(time_sample,x)
plt.legend(["N-clusters","N-iterations","Data Size %"],loc=4)
plt.xlabel('CPU time in sec ')
plt.ylabel('y')

fig.set_tight_layout(True)
plt.show()
fig.savefig('plots/AD_ModComp_Kmeans.pdf')


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
    clusterer = GaussianMixture(icomponents, n_init = 1, covariance_type='full',random_state=n_seed).fit(X) 
    end = time.time()
    BIC.append(clusterer.bic(X))
    AIC.append(clusterer.aic(X))
    time_components.append(end - start) 


# In[21]:

time_iter = []
time_sample = []
niter = ncomponents  
for iiter in range(1,niter):
    start = time.time() # iterations
    clusterer = GaussianMixture(n_components = 2, n_init=iiter, covariance_type='full',random_state=iiter).fit(X) 
    end = time.time()
    time_iter.append(end - start)    
    start = time.time() # samples
    isample = int(data_frac * iiter/100)
    clusterer = GaussianMixture(n_components = 2, n_init = 1, covariance_type='full',random_state=n_seed).fit(X[0:isample]) 
    end = time.time()
    time_sample.append(end - start)      


# In[22]:

x = np.arange(1, ncomponents)
sns.set_style("whitegrid")
fig = plt.figure(figsize=(11,4))

p1 = plt.subplot(121, title = 'AD EM Gaussian Mixture choosing n_components')
plt.plot(x, BIC, label='BIC')
plt.plot(x, AIC, label='AIC')
plt.axvline(x=11,color='magenta', linestyle='--', linewidth=0.5)
plt.legend(loc='best')
plt.xlabel('n_components')
plt.ylabel('y')

p2 = plt.subplot(122, title = 'AD EM Gaussian Mixture Model Complexity')
plt.plot(time_components,x, linestyle='--')
plt.plot(time_iter,x)
plt.plot(time_sample,x)
plt.legend(["N-components","N-iterations","Data Size %"],loc=4)
plt.xlabel('CPU time in sec ')
plt.ylabel('y')

fig.set_tight_layout(True)
plt.show()
fig.savefig('plots/AD_ModComp_EM.pdf')


# In[ ]:




# # Note!
# **Since training was not provided any information about target feature, it makes sence to see very close accuracy
# on both training and testing sets.**

# ### Apply PCA to Normalized Data 

# In[23]:

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
plt.title('AD Best Principal Components')

ax2 = fig.add_subplot(122)
aig_vals = pca_all.explained_variance_ratio_
y_pos = np.arange(len(aig_vals))
cumsum = np.cumsum(aig_vals)  
plt.bar(y_pos, aig_vals, align='edge', alpha=0.8, color='red', width=-0.3)
plt.bar(y_pos, cumsum, align='edge', alpha=0.8, width=0.3)
plt.legend(["Fraction of explained variance","Cumulative Sum for Explained variance fraction"],loc='best')
plt.ylabel('')
plt.xlabel('Principal Components') 
plt.title('AD Moddel Explained Var. by PCs')
plt.ylim(0,1.2)
fig.set_tight_layout(True)
fig.savefig('plots/AD_pca.pdf')


# ### Apply k-means to PCA output

# In[145]:

kmeans = KMeans(n_clusters=n_centroids, n_init=1, random_state=n_seed)
kmeans.fit(pca_n)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# In[159]:

pca_n1 = pd.DataFrame(pca_n)
pca_n1['new_tar'] = kmeans.labels_


# In[ ]:




# ### Apply EM to PCA output

# In[160]:

gm = GaussianMixture(n_components=n_centroids, n_init= 1, random_state=n_seed)
gm.fit(pca_n)
gm_labels = gm.predict(pca_n)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# In[161]:

pca_n2 = pd.DataFrame(pca_n)
pca_n2['new_tar'] = gm_labels


# ##  ICA

# In[29]:

ica =FastICA(algorithm='parallel', tol=0.001, whiten=True, fun='logcosh', max_iter=100, random_state=n_seed)
ica_all = ica.fit_transform(train)


# ### Applying k-means to ICA output

# In[30]:

kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(ica_all)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# In[ ]:




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
kurtosis_thresh = y.max()/3


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
plt.title('AD 3 Best Independent Components')

ax2 = fig.add_subplot(122)
plt.xlabel('Independent components', fontsize = 14)
plt.ylabel('Kurtosis', fontsize = 14)
plt.plot(x ,y, '-',lw=2., color='blue')
plt.axvline(x=kurt_max,color='magenta', linestyle='--', linewidth=0.5)
plt.axhline(y=kurtosis_thresh,color='magenta', linestyle='--', linewidth=0.5)
plt.title('AD Kurtosis plot')
fig.set_tight_layout(True)
fig.savefig('plots/AD_ica.pdf')


# ### Apply k-means to reduced dim by ICA

# In[162]:

kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(ica_keep)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# ### Apply EM to reduced dim by ICA

# In[163]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, max_iter=100)
gm.fit(ica_keep)
gm_labels = gm.predict(ica_keep)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# In[40]:

ica_n = ica_keep


# In[165]:

ica_n1 = pd.DataFrame(ica_n)
ica_n1['new_tar'] = kmeans.labels_


# In[166]:

ica_n2 = pd.DataFrame(ica_n)
ica_n2['new_tar'] = gm_labels


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


# In[52]:

fig = plt.figure(figsize=(5,4))
y = tmp.mean(axis=1)
x = np.arange(2,dims,2)
ax2 = fig.add_subplot(111)
plt.xlabel('Number of RP Components', fontsize = 14)
plt.ylabel('Reconstruction Error', fontsize = 14)
plt.plot(x ,y, '-o',lw=2., color='magenta')
plt.title('AD Reconstruction Error')
fig.set_tight_layout(True)
fig.savefig('plots/AD_rpa.pdf')


# In[45]:

rp_n = random_projection.GaussianRandomProjection(n_components=n_features,
                                                  eps=0.9,
                                                  random_state=n_seed).fit_transform(train)


# In[62]:

rp_n.shape


# In[167]:

kmeans = KMeans(n_clusters=n_centroids,random_state = n_seed)
kmeans.fit(rp_n)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# In[168]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, max_iter=100)
gm.fit(rp_n)
gm_labels = gm.predict(rp_n)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# In[169]:

rp_n1 = pd.DataFrame(rp_n)
rp_n1['new_tar'] = kmeans.labels_


# In[170]:

rp_n2 = pd.DataFrame(rp_n)
rp_n2['new_tar'] = gm_labels


# ## Factor Analysis

# In[47]:

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score


# In[48]:

fa = FactorAnalysis(random_state=n_seed)


# In[49]:

fa_scores = []
X=train[0:5000] # try it on smaller data subset
for n in np.arange(1,30):
    fa.n_components = n
    fa_scores.append(np.mean(cross_val_score(fa, X)))


# In[50]:

sns.set_style("whitegrid")
x, y = np.arange(0,len(fa_scores)), np.array(fa_scores)
score_max = x[y.argmax()]

fig = plt.figure(figsize=(5,4))
plt.plot(x, y)
plt.xlabel('Number of FA Components', fontsize = 14)
plt.ylabel('cv score', fontsize = 14)
plt.title('AD Factor analysis CV score')
plt.axvline(x=score_max,color='magenta', linestyle='--', linewidth=0.5)
fig.set_tight_layout(True)
fig.savefig('plots/AD_fa.pdf')


# In[80]:

fa_n = FactorAnalysis(n_components= score_max, random_state=n_seed).fit_transform(train)


# In[176]:

kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(fa_n)
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


# In[177]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, max_iter=100)
gm.fit(fa_n)
gm_labels = gm.predict(fa_n)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))


# In[178]:

fa_n1 = pd.DataFrame(fa_n)
fa_n1['new_tar'] = kmeans.labels_


# In[179]:

fa_n2 = pd.DataFrame(fa_n)
fa_n2['new_tar'] = gm_labels


# ## DR Benchmarking

# In[53]:

time_comp_pca = []
time_sample_pca = []
time_comp_ica = []
time_sample_ica = []
time_comp_rp = []
time_sample_rp = []
time_comp_fa = []
time_sample_fa = []
ncomponents = 60 
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
    


# In[60]:

x = np.arange(2, ncomponents)
sns.set_style("whitegrid")
fig = plt.figure(figsize=(6,4))

plt.subplot(111, title = 'AD Dim. Reduction Time Benchmarking')
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
fig.savefig('plots/AD_DR_benchmarks.pdf')


# ## ANN 

# In[61]:

from sklearn.neural_network import MLPClassifier


# In[94]:

MLP = MLPClassifier(hidden_layer_sizes=(5,),
                    activation='logistic',
                    solver='adam',
                    learning_rate_init=0.001,
                    max_iter=200,
                    shuffle=True,
                    random_state=n_seed)


# In[95]:

MLP.fit(train,train_labels)


# ## NN prediction on orig data

# In[96]:

nn_pred_train_ori = MLP.predict(train)


# In[97]:

print(confusion_matrix(train_labels,nn_pred_train_ori))
print(classification_report(train_labels,nn_pred_train_ori))


# In[123]:

MLP.fit(train,train_labels)
train_score_ori = MLP.score(train,train_labels)


# In[125]:

MLP.fit(pca_n,train_labels)
train_score_pca = MLP.score(pca_n,train_labels)


# In[118]:

MLP.fit(ica_n,train_labels)
train_score_ica = MLP.score(ica_n,train_labels)


# In[119]:

MLP.fit(rp_n,train_labels)
train_score_rp = MLP.score(rp_n,train_labels)


# In[120]:

MLP.fit(fa_n,train_labels)
train_score_fa = MLP.score(fa_n,train_labels)


# In[126]:

all_train_scores = [train_score_ori, train_score_pca, train_score_ica, train_score_rp, train_score_fa]


# In[127]:

all_train_scores


# ## Add K-means Predicted as new column

# In[174]:

kmeans = KMeans(n_clusters=n_centroids, n_init=1, n_jobs=4, random_state=n_seed)
kmeans.fit(train)

train_new= pd.DataFrame(train)
train_new['new_tar'] = kmeans.labels_


# In[180]:

MLP.fit(train_new,train_labels)
train_score_ori1 = MLP.score(train_new,train_labels)

MLP.fit(pca_n1,train_labels)
train_score_pca1 = MLP.score(pca_n1,train_labels)

MLP.fit(ica_n1,train_labels)
train_score_ica1 = MLP.score(ica_n1,train_labels)

MLP.fit(rp_n1,train_labels)
train_score_rp1 = MLP.score(rp_n1,train_labels)

MLP.fit(fa_n1,train_labels)
train_score_fa1 = MLP.score(fa_n1,train_labels)


# In[181]:

all_train_scores_kmeans = [train_score_ori1, train_score_pca1, train_score_ica1, train_score_rp1, train_score_fa1]


# In[182]:

all_train_scores_kmeans


# ## Add EM Predicted as new column

# In[183]:

gm = GaussianMixture(n_components=n_centroids, random_state=n_seed, n_init=1, max_iter=100)
gm.fit(train)
gm_labels = gm.predict(train)

train_new= pd.DataFrame(train)
train_new['new_tar'] = gm_labels


# In[184]:

MLP.fit(train_new,train_labels)
train_score_ori2 = MLP.score(train_new,train_labels)

MLP.fit(pca_n2,train_labels)
train_score_pca2 = MLP.score(pca_n2,train_labels)

MLP.fit(ica_n2,train_labels)
train_score_ica2 = MLP.score(ica_n2,train_labels)

MLP.fit(rp_n2,train_labels)
train_score_rp2 = MLP.score(rp_n2,train_labels)

MLP.fit(fa_n2,train_labels)
train_score_fa2 = MLP.score(fa_n2,train_labels)


# In[185]:

all_train_scores_em = [train_score_ori2, train_score_pca2, train_score_ica2, train_score_rp2, train_score_fa2]


# In[186]:

all_train_scores_em


# In[ ]:




# In[206]:

fig = plt.figure(figsize=(12,5))

ax=fig.add_subplot(111)
x=np.arange(0,5)
names=["lala","ori","pca","ica","rp","fa"]

plt.bar(x, all_train_scores, align='edge', alpha=0.8, color='red', width=-0.2)
plt.bar(x, all_train_scores_kmeans, align='edge', alpha=0.8, width=0.2)
plt.bar(x, all_train_scores_em, align='center', alpha=0.8, width=0.2)
plt.legend(["original","k-means added","em added"],loc='best')
plt.ylabel('Accuracy')
plt.xlabel('') 
plt.title('AD Prediction Accuracies')
plt.ylim(0,1.2)
ax.set_xticklabels(names)
fig.set_tight_layout(True)
fig.savefig('plots/all_acc.pdf')


# In[ ]:



