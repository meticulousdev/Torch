# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
  
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA


# %%
# X = pd.read_csv('./chap03/data/credit card.csv') 
X = pd.read_csv('./data/credit card.csv') 
X = X.drop('CUST_ID', axis = 1) 
X.fillna(method ='ffill', inplace = True)

print(X.head())

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TODO Gaussian distribution
X_normalized = normalize(X_scaled) 
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 

print(X_principal.head())

# %%
db_default = DBSCAN(eps = 0.0375, min_samples = 3).fit(X_principal)
labels = db_default.labels_

# %%
colours = {} 
colours[0] = 'y'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'
  
cvec = [colours[label] for label in labels]

r = plt.scatter(X_principal['P1'], X_principal['P2'], color ='y') 
g = plt.scatter(X_principal['P1'], X_principal['P2'], color ='g') 
b = plt.scatter(X_principal['P1'], X_principal['P2'], color ='b') 
k = plt.scatter(X_principal['P1'], X_principal['P2'], color ='k')
plt.close()

plt.figure(figsize=(7, 7)) 
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec) 
  
plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))
plt.show()

# %%
db = DBSCAN(eps = 0.0375, min_samples = 50).fit(X_principal) 
labels1 = db.labels_

colours1 = {} 
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'
  
cvec = [colours1[label] for label in labels1] 
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k' ] 
  
r = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[0]) 
g = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[1]) 
b = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[2]) 
c = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[3]) 
y = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[4]) 
m = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[5]) 
k = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[6])
plt.close()
  
plt.figure(figsize =(7, 7)) 
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec) 
plt.legend((r, g, b, c, y, m, k), 
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'), 
           scatterpoints = 1, 
           loc ='upper left', 
           ncol = 3, 
           fontsize = 8) 
plt.show()

# %%
db = DBSCAN(eps = 0.0375, min_samples = 100).fit(X_principal)
labels1 = db.labels_

colours1 = {} 
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'
  
cvec = [colours1[label] for label in labels1] 
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k' ] 
  
r = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[0]) 
g = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[1]) 
b = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[2]) 
c = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[3]) 
y = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[4]) 
m = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[5]) 
k = plt.scatter(X_principal['P1'], X_principal['P2'], 
                marker ='o', color = colors1[6]) 
plt.close()
  
plt.figure(figsize =(7, 7)) 
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec) 
plt.legend((r, g, b, c, y, m, k), 
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'), 
           scatterpoints = 1, 
           loc ='upper left', 
           ncol = 3, 
           fontsize = 8) 
plt.show()
