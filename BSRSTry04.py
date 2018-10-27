###Using logistic regression for BSRS predicting Suicidal Ideation
##Model 1
import pandas as pd
mydata=pd.read_csv('TotalBSRSnoMissing.csv',dtype=float)
import numpy as np
msk=np.random.rand(len(mydata))<0.8
train_data=mydata[msk]
test_data=mydata[~msk]

from sklearn.utils import shuffle
train_data=shuffle(train_data)
test_data=shuffle(test_data)

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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation

model=Sequential()
model.add(Dense(1,activation='sigmoid',input_dim=5))
model.add(Dropout(0.2))
#model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model_history=model.fit(train_x,train_y01,epochs=100,validation_split=0.2,batch_size=5,verbose=2)

import matplotlib.pyplot as plt
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.show()

prediction=model.predict_classes(test_x)
from sklearn.metrics import confusion_matrix as cms
mycms=cms(test_y01,prediction)
print(mycms)
from sklearn.metrics import accuracy_score
print (accuracy_score(test_y01,prediction))

##Model 2
# import pandas as pd
# mydata=pd.read_csv('TotalBSRSnoMissing.csv',dtype=float)
# import numpy as np
# msk=np.random.rand(len(mydata))<0.8
# train_data=mydata[msk]
# test_data=mydata[~msk]
# from sklearn.utils import shuffle
# train_data=shuffle(train_data)
# test_data=shuffle(test_data)
# train_x=train_data[['BSRS1','BSRS2','BSRS3','BSRS4','BSRS5']]
# train_y=train_data[['BSRS6']]
# #train_x=train_x.values.T.tolist()
# train_y=train_y.values
# train_y01=[0 if i <3.0 else 1 for i in train_y]
# train_y01=np.array(train_y01)
#
# test_x=test_data[['BSRS1','BSRS2','BSRS3','BSRS4','BSRS5']]
# test_y=test_data[['BSRS6']]
# test_y=test_y.values
# test_y01=[0 if i<3.0 else 1 for i in test_y ]
# test_y01=np.array(test_y01)
#
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers.core import Activation
# from keras import regularizers
# model=Sequential()
# model.add(Dense(1,activation='sigmoid',input_dim=5,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01)))
# model.add(Dropout(0.2))
# ##model.add(Activation('softmax'))
# model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
# model_history=model.fit(train_x,train_y01,epochs=10,validation_split=0.2,batch_size=5,verbose=2)
#
# import matplotlib.pyplot as plt
# plt.plot(model_history.history['acc'])
# plt.plot(model_history.history['val_acc'])
# plt.show()
#
# prediction=model.predict_classes(test_x)
# from sklearn.metrics import confusion_matrix as cms
# mycms=cms(test_y01,prediction)
# print(mycms)
# from sklearn.metrics import accuracy_score
# print (accuracy_score(test_y01,prediction))





