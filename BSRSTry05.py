###Using CNN for BSRS predicting Suicidal Ideation
##這是一個失敗的model,Training一直很高但Validation一直很差,怎麼都改善不了...

import pandas as pd
mydata=pd.read_csv('TotalBSRSnoMissing.csv',dtype=float)
import numpy as np
msk=np.random.rand(len(mydata))<0.8
train_data=mydata[msk]
test_data=mydata[~msk]
train_x=train_data[['BSRS1','BSRS2','BSRS3','BSRS4','BSRS5']]
train_y=train_data[['BSRS6']]
#train_x=train_x.values.T.tolist()
train_y=train_y.values
train_y01=[0 if i <3.0 else 1 for i in train_y]
train_y01=np.array(train_y01)

test_x=test_data[['BSRS1','BSRS2','BSRS3','BSRS4','BSRS5']]
test_y=test_data[['BSRS6']]
test_y=test_y.values
test_y01=[0 if i<3.0 else 1 for i in test_y ]
test_y01=np.array(test_y01)
from sklearn import preprocessing
minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
train_x=minmax_scale.fit_transform(train_x)
test_x=minmax_scale.fit_transform(test_x)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras import regularizers
from keras.layers.recurrent import SimpleRNN

model=Sequential()
model.add(Dense(units=4, input_dim=5, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=3,kernel_initializer='uniform',activation='relu',))
#model.add(Dense(2,activation='sigmoid',input_dim=5,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.2))
model.add(Dense(units=2,kernel_initializer='uniform',activation='relu'))
#model.add(Dense(units=1,activation='softmax'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
#model_history=model.fit(train_x,train_y01,epochs=30,validation_split=0.2,batch_size=5,verbose=2)
model_history=model.fit(train_x,train_y01,epochs=10,validation_split=0.2,batch_size=1,verbose=2)

import matplotlib.pyplot as plt
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.legend(['Train','Validation'],loc='best')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.legend(['Train','Validation'],loc='best')
plt.show()

prediction=model.predict_classes(test_x)
from sklearn.metrics import confusion_matrix as cms
mycms=cms(test_y01,prediction)
print(mycms)
from sklearn.metrics import accuracy_score
print (accuracy_score(test_y01,prediction))





