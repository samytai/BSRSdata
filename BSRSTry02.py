###20180211使用BSRS作x預測自殺意念Y
###先改變BSRST(x)的標準為15分,而不是10分...並計算最佳切點
import numpy as np
import sklearn as skn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cmx
import seaborn as sn

data=pd.read_csv('TotalBSRSnoMissing.csv',dtype=float, )
x=data[['BSRST']]
y=data[['BSRS6']]
x=x.values.T.tolist()
y=y.values.T.tolist()
x_array=np.asarray(x)
predictionX=[0 if i<15.0 else 1 for i in x_array.reshape(-1)]##由BSRS總分所推論的自殺意念
y_array=np.asarray(y)
predictionY=[0 if i<3.0 else 1 for i in y_array.reshape(-1)]###由第六題所推論的自殺意念
valueX=x[0]
valueY=y[0]
from sklearn.metrics import accuracy_score
print(accuracy_score(predictionY,predictionX))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(predictionY,x[0]))
from sklearn.metrics import roc_curve, auc
fpr,tpr,thr=(roc_curve(predictionY,x[0]))

##plt.xticks(thr[:-1])
sen_minus_spe=(tpr-1+fpr)
sen_plus_spe=(tpr+1-fpr)
plt.plot(sen_minus_spe[:-1],color='red', label=u'敏感度-專一度(越接近0越好)')
plt.plot(sen_plus_spe[:-1],color='blue',label=u'敏感度+專一度(越大越好)')
#plt.plot(tpr-1+fpr,color='red', label=u'敏感度-專一度(越接近0越好)')
#plt.plot(tpr+1-fpr,color='blue',label=u'敏感度+專一度(越大越好)')
##x_ticks=thr[:-1]


plt.xlabel(u'BSRS總分')
plt.legend(loc='best')
plt.axvline(x=14, color='g',linestyle='--')
plt.show()


i=np.arange(len(tpr))
roc=pd.DataFrame({'fpr':pd.Series(fpr, index=i),\
                'tpr':pd.Series(tpr,index=i),\
                '1-fpr':pd.Series(1-fpr,index=i),\
                't_minus_f':pd.Series((tpr-1+fpr),index=i),\
                't_plus_f':pd.Series((tpr+1-fpr),index=i),\
                'thresholds':pd.Series(thr,index=i)})
print(roc)
print(roc.ix[(roc.t_minus_f-0).abs().argsort()[:1]])
rroc=roc.iloc[::-1]
#print(rroc)
fig, ax = plt.subplots()
plt.plot(rroc[['thresholds']],rroc[['t_minus_f']],\
         color='red', label=u'敏感度-專一度(越接近0越好)')
plt.plot(rroc[['thresholds']],rroc[['t_plus_f']],\
         color='blue',label=u'敏感度+專一度(越大越好)')
plt.axvline(x=14,color='g',linestyle='--')
plt.legend(loc='best')
plt.show()





