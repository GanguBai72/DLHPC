#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[17]:


df= pd.read_csv("Boston.csv")


# In[18]:


df.head()


# In[19]:


df.drop(columns=['Unnamed: 15','Unnamed: 16'],inplace=True)


# In[20]:


df.drop(columns=['CAT. MEDV'],inplace=True)


# In[21]:


df.head()


# In[22]:


df.isnull().sum()


# In[23]:


df.info()


# In[24]:


df.corr()['MEDV'].sort_values()


# In[25]:


X = df.loc[:,['LSTAT','PTRATIO','RM']]
Y = df.loc[:,"MEDV"]
X.shape,Y.shape


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=10)


# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)


# In[28]:


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[29]:


from keras.models import Sequential
from keras.layers import Dense


# In[31]:


model = Sequential()


# In[32]:


model.add(Dense(128,input_shape=(3,),activation='relu',name='input'))
model.add(Dense(64,activation='relu',name='layer_1'))
model.add(Dense(1,activation='linear',name='output'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


# In[33]:


model.fit(x_train,y_train,epochs=100,validation_split=0.05)


# In[34]:


output = model.evaluate(x_test,y_test)


# In[35]:


print(f"Mean Squared Error: {output[0]}"
      ,f"Mean Absolute Error: {output[1]}",sep="\n")


# In[36]:


y_pred = model.predict(x=x_test)


# In[37]:


print(*zip(y_pred,y_test))


# In[42]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Actual vs. Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=14)
plt.ylabel('Predicted Price', fontsize=14)
plt.title('Actual vs. Predicted Housing Prices', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
     


# In[ ]:




