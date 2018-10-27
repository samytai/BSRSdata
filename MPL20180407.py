###使用MLP預測suicide ideation (>=3分)
import pandas as pd
#trainData=pd.read_csv('train_set.csv')
mydata=pd.read_csv('LogisticRegressionData20180408.csv')
mydata=mydata[['SuicidIdea','bsrs1','bsrs2','bsrs3','bsrs4','bsrs5','group']]
print(mydata.head())
trainData=mydata[mydata.group==1]
valData=mydata[mydata.group==2]
testData=mydata[mydata.group==3]
#print(len(trainData), len(valData),len(testData))

train_x=trainData[['bsrs1','bsrs2','bsrs3','bsrs4','bsrs5']]
train_y=trainData[['SuicidIdea']]

val_x=valData[['bsrs1','bsrs2','bsrs3','bsrs4','bsrs5']]
val_y=valData[['SuicidIdea']]

test_x=testData[['bsrs1','bsrs2','bsrs3','bsrs4','bsrs5']]
test_y=testData[['SuicidIdea']]

from sklearn import preprocessing
minmax_scale=preprocessing.MinMaxScaler(feature_range=(-1,1))
train_x=minmax_scale.fit_transform(train_x)
val_x=minmax_scale.fit_transform(val_x)
test_x=minmax_scale.fit_transform(test_x)

from keras.models import Sequential
from keras.layers import Dense, Dropout

model=Sequential()
model.add(Dense(units=1000, input_dim=5,kernel_initializer='normal',activation='relu'))
#model.add(Dropout(0.999))
model.add(Dense(units=5000,kernel_initializer='normal',activation='relu'))
#model.add(Dropout(0.999))
model.add(Dense(units=500,kernel_initializer='normal',activation='relu'))
#model.add(Dropout(0.999))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#train_history=model.fit(x=train_x,y=train_y,validation_split=0.1,epochs=30,batch_size=30,verbose=2)
# from keras.callbacks import EarlyStopping
# myEarlyStopping=EarlyStopping(monitor='val_acc',mode='auto')
# train_history=model.fit(x=train_x,y=train_y,epochs=100,batch_size=500,verbose=2,
#                         validation_data=(val_x,val_y),callbacks=[myEarlyStopping])
train_history=model.fit(x=train_x,y=train_y,epochs=10,batch_size=500,verbose=2,
                        validation_data=(val_x,val_y))

score=model.evaluate(test_x,test_y,batch_size=500)
print(score)
#model.save('MLP20180408.h5')
import matplotlib.pyplot as plt
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.legend(['train_loss','val_loss','train_acc','val_acc'],loc='best')
plt.show()

# from keras.models import load_model
# mymodel=load_model('MLP20180408.h5')
mypredic=model.predict(test_x)
predicClass=model.predict_classes(test_x)
cb=pd.crosstab(predicClass.reshape(-1),test_y.values.reshape(-1))
print(cb)

