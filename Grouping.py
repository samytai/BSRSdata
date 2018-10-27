####區分所有資料為60%訓練組,20%測試組,20%驗證組
import pandas as pd
mydata=pd.read_csv('TotalBSRSnoMissing.csv')
mydata=mydata[['BSRS6','BSRS1','BSRS2','BSRS3','BSRS4','BSRS5','sex','age']]
from sklearn.utils import shuffle
mydata=shuffle(mydata)
import numpy as np
np.random.seed(10)
msk1=np.random.rand(len(mydata))<0.8
train_data=mydata[msk1]
test_set=mydata[~msk1]
mak2=np.random.rand(len(train_data))<0.75
train_set=train_data[mak2]
val_set=train_data[~mak2]
train_set.to_csv('train_set.csv',index=False, sep='\t', encoding='utf-8')
test_set.to_csv('test_set.csv',index=False, sep='\t', encoding='utf-8')
val_set.to_csv('val_set.csv',index=False, sep='\t',encoding='utf-8')
print(len(test_set), len(val_set), len(train_set))
