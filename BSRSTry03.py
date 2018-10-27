###20180307##########
import numpy as np
import sklearn as skn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cmx
import seaborn as sn

data=pd.read_csv('TotalBSRSnoMissing.csv',dtype=float)
x=data[['BSRST']]
y=data[['BSRS6']]
x=x.values.T.tolist()
y=y.values.T.tolist()
###cm=pd.crosstab(y,x)
###print(cm)
###BSRST(x)大於14分者為positive predictionX, X_15
###BSRS6(y)大於2者為positive predictionY, Y_3
xarray=np.asarray(x)
X_15=[0 if i<14.0 else 1 for i in xarray.reshape(-1)]
X_10=[0 if i<9.0 else 1 for i in xarray.reshape(-1)]
yarray=np.asarray(y)
Y_4=[0 if i<3.0 else 1 for i in yarray.reshape(-1)]
Y_3=[0 if i<2.0 else 1 for i in yarray.reshape(-1)]
Y_2=[0 if i<1.0 else 1 for i in yarray.reshape(-1)]
Y_1=[0 if i<0.0 else 1 for i in yarray.reshape(-1)]

from sklearn.metrics import accuracy_score
print('Y_4,X_15',accuracy_score(Y_4,X_15))
print('Y_3,X_15',accuracy_score(Y_3,X_15))
print('Y_2,X_15',accuracy_score(Y_2,X_15))
print('Y_1,X_15',accuracy_score(Y_1,X_15))
print('Y_4,X_10',accuracy_score(Y_4,X_10))
print('Y_3,X_10',accuracy_score(Y_3,X_10))
print('Y_2,X_10',accuracy_score(Y_2,X_10))
print('Y_1,X_10',accuracy_score(Y_1,X_10))

###畫ROC曲線
from sklearn.metrics import roc_curve, auc
fpr, tpr, thr=(roc_curve(Y_4 , x[0]))
print(auc(fpr,tpr))
fpr, tpr, thr=(roc_curve(Y_1 , x[0]))
print(auc(fpr,tpr))