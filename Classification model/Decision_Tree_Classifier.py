#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd 
import seaborn as sn
import numpy as np
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("C:\\Users\\Nakesh\\Desktop\\Machine Larning-P1\\Classification model\\combined & cleaned.csv")


# In[2]:


# Loading weather data 
weather  = data
print(weather.keys())
# ['Unnamed: 0', 'Date', 'Max.', 'Min.', 'Morn', 'Eve', 'Rain','Rainy Days ', 'PE', 'Year']


# In[3]:


df = pd.DataFrame(weather)
df = df.drop(columns=['Unnamed: 0' , 'Year'])

df['Rainy Days '].unique() # contains values that are wrong
# Replace incorrect categories
df['Rainy Days '] = df['Rainy Days '].replace({20.0: 0, 11.0: 1})
df['Rainy Days '].unique()
df


# In[4]:


df['Date'] = pd.to_datetime(df['Date'] , format= "%d.%m.%y")

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day


df = df.drop('Date' , axis=1)
df.isna().sum()


# In[5]:


df


# In[6]:


# filling Median values in the place of NaN values for column 

m_min = df['Max.'].median()
m_morn = df['Morn'].median()
m_Eve = df['Eve'].median()
m_PE = df['PE'].median()

df['Max.'] = df['Max.'].fillna(m_min)
df['Morn'] = df['Morn'].fillna(m_morn)
df['Eve'] = df['Eve'].fillna(m_Eve)
df['PE'] = df['PE'].fillna(m_PE)

df.isna().sum()

df.to_csv('ready_to_use_data.csv', index=False)
df


# In[7]:


print(df.columns)


# In[8]:


x = df.drop('Rainy Days ' , axis=1)
y = df['Rainy Days ']


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[10]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train , y_train)


# In[11]:


y_predicted = dtc.predict(x_test)
dtc.score(x_test , y_test)


# In[12]:


plt.figure(figsize=(50, 50))
plot_tree(
    dtc, 
    filled=True,
    feature_names=weather.drop('Rainy Days ', axis=1).columns,
    class_names=weather['Rainy Days '].unique().astype(str)
)
plt.show()


# In[13]:


cmetrix = confusion_matrix(y_test , y_predicted) 
cmetrix


# In[14]:


len(y_predicted)


# In[15]:


# visualizing it with seborn library 
plt.figure(figsize=(7,5))
sn.heatmap(cmetrix , annot = True , fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[25]:


print(f"Model predictiion was {accuracy_score(y_test, y_predicted, normalize=True, sample_weight=None)*100}% Accurate")


# In[ ]:




