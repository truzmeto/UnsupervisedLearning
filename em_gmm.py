import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.mixture import GaussianMixture               
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

gm = GaussianMixture(n_components=n_centroids,
                     random_state=n_seed,
                     n_init=1,
                     max_iter=100)
gm.fit(train)
gm_labels = gm.predict(train)
print(confusion_matrix(train_labels, gm_labels))
print(classification_report(train_labels, gm_labels))
test_gm = gm.predict(test)
print(confusion_matrix(test_labels, test_gm))
print(classification_report(test_labels, test_gm))

data_frac = 5000
X = train 
X = X[0:data_frac]
y = train_labels

## EM model Complexity + Choosing n-components
BIC = []
AIC = []
time_components = []
ncomponents = 50
for icomponents in range(1,ncomponents):    
    start = time.time() # components
    clusterer = GaussianMixture(icomponents,
                                n_init = 1,
                                covariance_type='full',
                                random_state=n_seed).fit(X)   
    end = time.time()    
    BIC.append(clusterer.bic(X))    
    AIC.append(clusterer.aic(X))    
    time_components.append(end - start)    

    
time_iter = []                                               
time_sample = []                                             
niter = ncomponents                                          
for iiter in range(1,niter):                                 
    start = time.time() # iterations                         
    clusterer = GaussianMixture(n_components = 2,
                                n_init=iiter,
                                covariance_type='full',
                                random_state=iiter).fit(X)
    end = time.time()                                                                              
    time_iter.append(end - start)                                                                  
    start = time.time() # samples                                                                  
    isample = int(data_frac * iiter/100)                                                           
    clusterer = GaussianMixture(n_components = 2,
                                n_init = 1,
                                covariance_type='full',
                                random_state=n_seed).fit(X[0:isample])
    end = time.time()                                                                              
    time_sample.append(end - start)                                                                

## visualize AIC BIC and model complexity    
x = np.arange(1, ncomponents)                                                                      
sns.set_style("whitegrid")                                                                         
fig = plt.figure(figsize=(11,4))                                                                   
p1 = plt.subplot(121, title = 'EM Gaussian Mixture choosing n_components')                         
plt.plot(x, BIC, label='BIC')                                                                      
plt.plot(x, AIC, label='AIC')                                                                      
plt.axvline(x=11,color='magenta', linestyle='--', linewidth=0.5)                                   
plt.legend(loc='best')                                                                             
plt.xlabel('n_components')                                                                         
plt.ylabel('y')

p2 = plt.subplot(122, title = 'EM Gaussian Mixture Model Complexity')
plt.plot(time_components,x, linestyle='--')
plt.plot(time_iter,x)
plt.plot(time_sample,x)
plt.legend(["N-components","N-iterations","Data Size %"],loc=4)
plt.xlabel('CPU time in sec ')
plt.ylabel('y')

fig.set_tight_layout(True)
plt.show()
fig.savefig('plots/ModComp_EM.pdf')
