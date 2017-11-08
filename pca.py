import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import sys

# pass path+filenames as argument for processing
train_path = sys.argv[1] # pass training data
test_path =  sys.argv[2] # pass testing data

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



# when n_comp is given as fraction and svd solver is full, then algorithm chooses n_components
# such that explained variance in a model = n_components=0.99 here
pca_all = PCA(n_components=0.99,  svd_solver = 'full')
pca_all.fit(train)
pca_n = pca_all.transform(train)


# visualize 1st 3 components + explained variance
sns.set_style("whitegrid")
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pca_n[:,2], pca_n[:,0],pca_n[:,1],c=train_labels,cmap='plasma')
ax1.set_xlabel('PC2')
ax1.set_ylabel('PC0')
ax1.set_zlabel('PC1')
plt.title('Best Principal Components')

ax2 = fig.add_subplot(122)
aig_vals = pca_all.explained_variance_ratio_
y_pos = np.arange(len(aig_vals))
cumsum = np.cumsum(aig_vals)  
plt.bar(y_pos, aig_vals, align='edge', alpha=0.8, color='red', width=-0.3)
plt.bar(y_pos, cumsum, align='edge', alpha=0.8, width=0.3)
plt.legend(["Fraction of explained variance","Cumulative Sum for Explained variance fraction"],loc='best')
plt.ylabel('')
plt.xlabel('Principal Components') 
plt.title('Moddel Explained Var. by PCs')
plt.ylim(0,1.2)
fig.set_tight_layout(True)
fig.savefig('plots/pca.pdf')


## Apply k-means to PCA output
kmeans = KMeans(n_clusters=n_centroids, n_init=1, random_state=n_seed)
kmeans.fit(pca_n)
print("k-means prediction results for pca output are")
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


## Apply EM to PCA output
gm = GaussianMixture(n_components=n_centroids, n_init= 1, random_state=n_seed)
gm.fit(pca_n)
gm_labels = gm.predict(pca_n)

print("EM Gaussian Mixture Model prediction results for pca output are")
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))

