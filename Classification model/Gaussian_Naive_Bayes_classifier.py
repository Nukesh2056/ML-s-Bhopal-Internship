#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,accuracy_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sn
weather = pd.read_csv('ready_to_use_data.csv')


# In[2]:


df = pd.DataFrame( weather )
df


# In[3]:


x = df.drop('Rainy Days ' , axis=1)
y = df['Rainy Days ']


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=42)


# In[5]:


model = GaussianNB()
model.fit(x_train , y_train)


# In[6]:


y_predicted = model.predict(x_test)


# In[7]:


cmetrix = confusion_matrix(y_test , y_predicted) 
cmetrix


# In[8]:


# visualizing it with seborn library 
plt.figure(figsize=(7,5))
sn.heatmap(cmetrix , annot = True , fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[11]:


print(f"Accuracy Score : {accuracy_score(y_test , y_predicted)*100}%")


# In[ ]:





# In[ ]:




