---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Predicting Breast Cancer Dataset using PCA

* Use PCA to narrow down breast-cancer dataset to two principal components
    - For Visualisation
    - For comparisonin in performance & accuracy compared to direct methods
    
* Use SVM based classifier for creating Malignant/Benign Classifier
    - Grid Search for optimal parameter selection
    - Imapct of PCA on classifier accuracy

```python
import numpy as np 
import pandas as pd 
```

```python
import matplotlib.pyplot as plt 

%matplotlib notebook 
```

```python
plt.style.use('seaborn')
```

```python
from sklearn.datasets import load_breast_cancer
```

```python
data = load_breast_cancer()
```

```python
data.keys()
```

```python
np.atleast_2d( data['target']).T.shape
```

```python
np.c_[data['data'],data['target']].shape
```

```python
np.append(data['feature_names'],'target') 
```

```python
df = pd.DataFrame(data = np.c_[data['data'],data['target']],
                  columns = np.append(data['feature_names'],'target') 
                 )
df 
```

```python
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
```

```python
X
```

```python
y
```

```python
from sklearn.preprocessing import StandardScaler
```

```python
X_norm = StandardScaler().fit_transform(X)
X_norm 
```

```python
from sklearn.decomposition import PCA
```

```python
pca = PCA(n_components=6,svd_solver = 'auto')
pca
```

```python
principal_components = pca.fit_transform(X_norm)
```

```python
y.values.shape
```

```python
pc_df = pd.DataFrame(np.c_[principal_components,y.values],columns=[
    'PC1','PC2','PC3','PC4','PC5','PC6','target'
])
pc_df
```

```python
ax = pc_df.loc[pc_df['target']==0,['PC1','PC2']].plot(x = 'PC1',y = 'PC2',
                                                    kind = 'scatter',
                                                    c = 'red',
                                                   label = '$y=0$')
```

```python
pc_df.loc[pc_df['target']==1,['PC1','PC2']].plot(x = 'PC1',y = 'PC2',
                                                    kind = 'scatter',
                                                    c = 'green',
                                                   label = 'y=1',
                                                  ax = ax)
```

```python
plt.title('Data distribution for $PC_1$ and $PC_2$')
plt.gcf()
```

```python
np.arange(pca.n_components+1)
```

```python
plt.figure()
x = np.arange(1,pca.n_components+1)
plt.bar(x,pca.explained_variance_ratio_)
```

```python
plt.xlabel('Principal Components')
plt.ylabel('Proportion of $\sigma^2$ explained')
```

```python
plt.title('Scree Plot & Cumulative $\sigma^2$',fontsize = 20)
```

```python
ax = plt.gca()
ax2 = ax.twinx()
```

```python
ax2.plot(x,np.cumsum(pca.explained_variance_ratio_),    
        'r-',label = 'cumulative $\sigma^2$')
```

```python
ax2.axes.set_ylabel('Cumulative $\sigma^2$')
```

```python
ax2.set_ylim([0,1])
```

```python
ax2.plot(x,[0.7]*len(x),'g--',label='Threshold $\sigma^2 = 0.7$')
```

```python
plt.legend(loc = 'upper right')

```

```python
plt.gcf()
```

```python
pca.explained_variance_
```

```python
imp_features = pd.DataFrame(pca.components_.T,
                            columns = pc_df.columns[:-1],
                           index = df.columns[:-1])
imp_features
```

```python
imp_features.sort_values(by=['PC1','PC2','PC3'],ascending=False)
```

```python

```
