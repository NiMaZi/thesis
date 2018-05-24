import sys
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Bidirectional,Masking,BatchNormalization
from keras.callbacks import EarlyStopping

dim=128
maxlen=int(sys.argv[1])
batch_size=int(sys.argv[2])

model=Sequential()
model.add(Masking(mask_value=0.0,input_shape=(maxlen,dim)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(dim,return_sequences=False,dropout=0.5,activation="relu"),merge_mode='ave'))
model.compile(optimizer='nadam',loss='binary_crossentropy')

sample=1024
X=np.random.rand(sample,maxlen,dim)
y=np.random.rand(sample,dim)
X[:,3,:]=0.0
X[5,5,:]=0.0
X[1,5,:]=0.0


early_stopping=EarlyStopping(monitor='loss',patience=10)
early_stopping_val=EarlyStopping(monitor='val_loss',patience=10)
model.fit(X,y,batch_size=batch_size,epochs=10,shuffle=True,validation_split=0.1,callbacks=[early_stopping,early_stopping_val])

path="/home/yzg550/temp.h5"
model.save(path)


# import numpy as np
# from keras import optimizers
# from keras.models import Sequential,load_model
# from keras.layers import Dense,Dropout,BatchNormalization
# from keras.callbacks import EarlyStopping

# dim=8192
# hrate=1.5
# drate=0.5

# myAct='relu'
# model=Sequential()
# model.add(Dense(int(dim*hrate),input_dim=dim,activation=myAct))
# model.add(BatchNormalization())
# model.add(Dropout(drate))
# model.add(Dense(dim,activation=myAct))
# model.compile(optimizer='Nadam',loss='binary_crossentropy')

# sample=1024
# X=np.random.rand(sample,dim)
# y=np.random.rand(sample,dim)

# early_stopping=EarlyStopping(monitor='loss',patience=10)
# model.fit(X,y,batch_size=256,epochs=100,shuffle=True,callbacks=[early_stopping])