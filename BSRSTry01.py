##Try at 20190420
import numpy as np
import sklearn as skn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cmx
import seaborn as sn

data=pd.read_csv('TotalBSRSnoMissing.csv',dtype=float, )
#print(data.head())
x=data[['BSRST']]
y=data[['BSRS6']]
#cm=cmx(x,y)
x=x.values.T.tolist()
y=y.values.T.tolist()

# cm=pd.crosstab(y,x)
# print(cm)
# mpl.rcParams['font.sans-serif']=[u'MicrosoftJhungHeiRegular']
# HP=sn.heatmap(pd.crosstab(x,y),cmap='YlOrRd',annot=True,fmt='d',annot_kws={"size":8})
# HP.set(title='BSRS量表總分與自殺意念程度比較\n N=5,211',xlabel='自殺意念',ylabel='BSRS量表總分')
# xlist=['完全沒有','輕度','中度','重度','極重度']
# HP.set_xticklabels(xlist, minor=False)
# HP.invert_yaxis() ##加了這一段y軸才會由大到小排列
# plt.axhline(y=15,color='r',linestyle='--')
# plt.axvline(x=3,color='r',linestyle='--')
# plt.show()


x_array=np.asarray(x)
predictionX=[0 if i<10.0 else 1 for i in x_array.reshape(-1)]##由BSRS總分所推論的自殺意念

y_array=np.asarray(y)
predictionY=[0 if i<3.0 else 1 for i in y_array.reshape(-1)]###由第六題所推論的自殺意念
valueX=x[0]
ct0=pd.crosstab(y,np.asarray(predictionX))
print(ct0)
# '''
#ct1=pd.crosstab(np.asarray(predictionX),np.asarray(valueX))
# hm1=sn.heatmap(ct1, annot=True)
# plt.show()
# '''
#valueY=y[0]
# '''
# ct2=pd.crosstab(np.asarray(predictionY),np.asarray(valueY))
# hm2=sn.heatmap(ct2, annot=True)
# plt.show()
# '''
# from sklearn.metrics import accuracy_score
# print(accuracy_score(predictionY,predictionX))
# from sklearn.metrics import roc_auc_score
# print(roc_auc_score(predictionY,x[0]))
# from sklearn.metrics import roc_curve, auc
# fpr,tpr,thr=(roc_curve(predictionY,x[0]))
# roc_auc=auc(fpr, tpr)
# plt.plot(fpr,tpr,label='AUC = %0.2f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.01,1.01])
# plt.ylim([-0.01,1.01])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
#
# plt.show()





