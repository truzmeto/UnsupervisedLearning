import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from mpl_toolkits.mplot3d import Axes3D
import sys

# pass path+filenames as argument for processing
train_path = sys.argv[1]

# load data
train = pd.read_csv(train_path ,index_col=False)

train_labels = train['income'].values
train = train.drop('income',axis=1).values


#parameters
n_centroids = 2
n_seed = 100

#normalize everything
a = train
train = (a - a.mean()) / np.std(a)


##  ICA
ica =FastICA(algorithm='parallel',
             tol=0.001,
             whiten=True,
             fun='logcosh',
             max_iter=100,
             random_state=n_seed)
ica_all = ica.fit_transform(train)



# calculate kurtosis
y = kurtosis(ica_all, fisher=True)
x = np.arange(len(y)) 
kurt_max = x[y.argmax()]
kurtosis_thresh = y.max()/3


# ### Visualize ICA components with highest Kurtosis
tmp = [y > kurtosis_thresh]
indx = np.where(tmp)[1]
ica_all = pd.DataFrame(ica_all)
ica_keep = ica_all[indx]
print("ICA components to keep are:")
print(indx)

fig = plt.figure(figsize=(13,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(ica_keep[indx[0]], ica_keep[indx[1]],
            ica_keep[indx[2]],c=train_labels,cmap='plasma')
ax1.set_xlabel(r'$IC_0$')
ax1.set_ylabel(r'$IC_1$')
ax1.set_zlabel(r'$IC_2$')
plt.title('AD 3 Best Independent Components')

ax2 = fig.add_subplot(122)
plt.xlabel('Independent components', fontsize = 14)
plt.ylabel('Kurtosis', fontsize = 14)
plt.plot(x ,y, '-',lw=2., color='blue')
plt.axvline(x=kurt_max,color='magenta', linestyle='--', linewidth=0.5)
plt.axhline(y=kurtosis_thresh,color='magenta', linestyle='--', linewidth=0.5)
plt.title('AD Kurtosis plot')
fig.set_tight_layout(True)
plt.show()
#fig.savefig('plots/AD_ica.pdf')


## Apply k-means to reduced dim by ICA
kmeans = KMeans(n_clusters=n_centroids, random_state = n_seed)
kmeans.fit(ica_keep)
print("k-means prediction results for ICA output are")
print(confusion_matrix(train_labels,kmeans.labels_))
print(classification_report(train_labels,kmeans.labels_))


## Apply EM to reduced dim by ICA
gm = GaussianMixture(n_components=n_centroids,
                     random_state=n_seed,
                     max_iter=100)
gm.fit(ica_keep)
gm_labels = gm.predict(ica_keep)
print("EM Gaussian Mixture Model prediction results for pca output are")
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))

