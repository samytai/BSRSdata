###發現光用BSRS五題來預測suicidal ideation可能不夠..
##1.加入age, sex
##2.重新檢視到底使用random分類後,有沒有問題..
##3.直接用python的logistic regression看看ROC如何
##參照書本鐵達尼號的方式操作...
import pandas as pd
mydata=pd.read_csv('TotalBSRSnoMissing.csv')
mydata=mydata[['BSRS6','BSRS1','BSRS2','BSRS3','BSRS4','BSRS5','sex','age']]
import numpy as np
np.random.seed(10)###不知要幹什麼..書本上的...
msk=np.random.rand(len(mydata))<0.8
train_data=mydata[msk]
test_data=mydata[~msk]
#print('total:',len(mydata),'\ntrain:',len(train_data),'\ntest:',len(test_data))
train_y=train_data[['BSRS6']]
train_y=train_y.values
train_y=[0 if i<3 else 1 for i in train_y]
train_x=train_data.drop(['BSRS6'],axis=1)
train_y=np.asarray(train_y)

test_y=test_data[['BSRS6']]
test_y=test_y.values
test_y=[0 if i<3 else 1 for i in test_y]
test_y=np.asarray(test_y)
test_x=test_data.drop(['BSRS6'],axis=1)


from sklearn import preprocessing
minmax_scale=preprocessing.MinMaxScaler(feature_range=(-1,1))
train_x=minmax_scale.fit_transform(train_x)
test_x=minmax_scale.fit_transform(test_x)
#print(pd.DataFrame(train_y).tail())
#print(pd.DataFrame(train_x).tail())
print(type(train_y),type(train_x))
from keras.models import Sequential
from keras.layers import Dense, Dropout

model=Sequential()
model.add(Dense(units=10, input_dim=7,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=20,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
##train_history=model.fit(x=train_x,y=train_y,validation_split=0.1,epochs=30,batch_size=30,verbose=2)
train_history=model.fit(x=train_x,y=train_y,epochs=100,batch_size=3,verbose=2,validation_data=(test_x,test_y))

import matplotlib.pyplot as plt
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.legend(['train_loss','val_loss','train_acc','val_acc'],loc='best')
plt.show()
