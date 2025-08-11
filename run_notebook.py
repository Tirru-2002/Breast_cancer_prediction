import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
col_names = list(cancer.feature_names)
col_names.append('target')
df = pd.DataFrame(np.c_[cancer.data,cancer.target],columns = col_names)
df.head()
print(cancer.target_names)
df.info()
sns.pairplot(df)
X = df.drop('target',axis = 1)
y = df.target
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=55)
X_train
y_train
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
print(X_train[:,:])
from sklearn.ensemble import RandomForestClassifier
lin_reg = RandomForestClassifier()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,y_pred)
print('confusion matrix : \n',cm)
y_pred
y_test
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print("Accuracy:",round(metrics.accuracy_score(y_test,y_pred),2))
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X)
X_pca=pca.transform(X)
X.shape
X_pca.shape
y.shape
X_pca
plt.figure(figsize = (8,6))
sns.scatterplot(x = X_pca[:,0],y = X_pca[:,1],hue = y)
plt.xlabel("first pc")
plt.ylabel("second pc")
from sklearn.ensemble import RandomForestClassifier
logs = RandomForestClassifier()
logs.fit(X_train,y_train)
y_pred = logs.predict(X_test)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,y_pred)
print('confusion matrix : \n',cm)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print("Accuracy:",round(metrics.accuracy_score(y_test,y_pred),2))
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=55), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
y_pred_grid = grid_search.predict(X_test)
print("Accuracy with tuned parameters: ", metrics.accuracy_score(y_test, y_pred_grid))
