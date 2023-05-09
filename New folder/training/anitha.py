#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


Data = pd.read_csv('C:/Users/ALEX/Desktop/intelligent admission The future of university decision making with machine learning/Dataset/Admission_Predict.csv')
Data


# In[12]:


Data.info()


# In[33]:


Data.isnull().any()


# In[50]:


Data=Data.rename(columns ={'Chance of Admit ':'Chance of Admit'})


# In[35]:


Data.describe()


# In[36]:


sns.distplot(Data['GRE Score'])


# In[37]:


sns.pairplot(data=Data, hue='Research',markers=["^", "v"],palette='inferno')


# In[38]:


sns.scatterplot(x='University Rating',y='CGPA',data=Data, color='Red',s=100)


# In[49]:


category = ['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research','Chance of Admit']
color = ['yellowgreen','gold','lightskyblue','pink','red','purple','orange','gray']
start = True
for i in np.arange(4):
    fig = plt.figure(figsize=(14,8))
    plt.subplot2grid((4,2),(i,0))
    Data[category[2*i]].hist(color=color[2*i],bins=10)
    plt.title(category[2*i])
    plt.subplot2grid((4,2),(i,1))
    Data[category[2*i+1]].hist(color=color[2*i+1],bins=10)
    plt.title(category[2*i+1])
    
plt.subplots_adjust(hspace = 0.7,wspace = 0.2)
plt.show()
    


# In[43]:


x = Data.iloc[:,0:7].values
x


# In[44]:


y=Data.iloc[:,7:].values
y


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30,random_state=101)


# In[48]:


y_train=(y_train>0.5)
y_train


# In[ ]:


import tensorflow as tf
fromtensorflow import keras
from tensorflow.keras.layers import Dense, Activation, 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




