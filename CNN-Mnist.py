#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q keras')


# In[ ]:



import keras

from keras.datasets import mnist


# In[ ]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[ ]:


print(x_train.shape)


# In[ ]:


y_train


# In[ ]:



import pandas as pd
import numpy as np
import cv2


# In[ ]:


from google.colab.patches import cv2_imshow
for i in range(0,9):
  random_num=np.random.randint(0,len(x_train))
  img=x_train[random_num]
  window_name='Random sample #'+str(i)
  cv2_imshow(img)
  cv2.waitKey(3)

cv2.destroyAllWindows()


# In[ ]:


img_rows=x_train[0].shape[0]
img_cols=x_train[1].shape[0]


# In[ ]:


print(img_rows)


# In[ ]:


x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)


# In[ ]:


input_shape=(img_rows,img_cols,1)


# In[ ]:


x_train=x_train.astype('float32')
x_test=x_test.astype('float32')


# In[ ]:


x_train /= 255
x_test /= 225


# In[ ]:


x_train


# In[ ]:


from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

num_classes=y_test.shape[1]
num_pixels=y_train.shape[1] * x_train.shape[2]


# In[ ]:


y_train


# In[ ]:


y_test


# In[ ]:


num_classes


# In[ ]:


num_pixels


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer=SGD(0.01),metrics=['accuracy'])


# In[ ]:


model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test))


# In[ ]:


score=model.evaluate(x_test,y_test,verbose=0)
print('test loss',score[0])
print('test acc',score[1])


# In[ ]:


model.save('MNIST-weights.h5')


# In[ ]:


i=9
import matplotlib.pyplot as plt
predicted=model.predict(x_test)
while i <= 9:
    pred=x_test[i][:,:,0]

    g = plt.imshow(pred)
    pre=x_test[i]
    pre=pre.reshape(1,28,28,1)
    res=str(model.predict_classes(pre,1,verbose=0))
    print(res)
    i += 1
    
    


# In[ ]:




