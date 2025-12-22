#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import xgboost
from xgboost import XGBClassifier , XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from xgboost.callback import EarlyStopping

data = pd.read_csv('combined_weather_data.csv')
data = data.drop('cloud cover' , axis=1)
data


# In[2]:


# Convert the column to numeric
column_to_convert = ['Max.' , 'Min.' , 'Morn' , 'Eve' , 'Rain', 'Rainy Days ', 'PE']
# data[column_to_convert] = pd.to_numeric(data[column_to_convert], errors='coerce')
data[column_to_convert] = data[column_to_convert].apply(pd.to_numeric, errors='coerce')
data


# In[3]:


data.shape


# In[4]:


data.isna().sum()


# In[5]:


data.dropna(subset=['Rainy Days '] , inplace=True)


# In[6]:


data.isna().sum()


# In[7]:


data['Max.'] = data['Max.'].fillna(data['Max.'].median())
data['Morn'] = data['Morn'].fillna(data['Morn'].median())
data['Eve'] = data['Eve'].fillna(data['Eve'].median())
data['PE'] = data['PE'].fillna(data['PE'].median())


# In[8]:


data.isna().sum() # data is now cleaned 


# # Model Traning --> XG Boost Classifier for Rain Occurance

# In[9]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay , r2_score , mean_absolute_error , mean_squared_error , root_mean_squared_error
import matplotlib.pyplot as plt 
import numpy as np
model = XGBClassifier


# In[10]:


plt.figure(figsize=(10, 6))

plt.plot(data['Date'], data['Max.'], marker='o', linestyle='-', label='Max Temperature')
plt.plot(data['Date'], data['Min.'], marker='o', linestyle='-', label='Min Temperature')

plt.title('Max and Min Daily Temperatures')
plt.xlabel('Date') # X-axis label
plt.ylabel('Temperature (°C)') # Y-axis label
plt.legend() # Show legend to identify lines
plt.grid(True) # grid for better readability

# 4. Show the plot
plt.show()


# In[11]:


data['Date'] = pd.to_datetime(data['Date'] , format='%d.%m.%y')

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

data = data.drop(columns=['Date'])


# Keep only rows where label is 0 or 1
data = data[data['Rainy Days '].isin([0, 1])]


# In[12]:


data['Rainy Days '].unique()


# In[13]:


X = data.drop(columns=['Rainy Days ' , 'Rain'])
y = data['Rainy Days ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[14]:


y_train.value_counts()


# In[15]:


y_test.value_counts()


# In[16]:


2606/598


# In[17]:


651/150


# In[18]:


model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss' , max_depth = 19, n_estimators = 100)  # disable label encoder warning
model.fit(X_train, y_train)


# In[19]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)


# In[20]:


cm = confusion_matrix(y_test , y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# # Model Traning of XG Boost Regressor for Rainfall Amount

# In[106]:


modelR = XGBRegressor(n_estimators = 2300 ,  gamma = 8  , device = 'cuda' ,max_dept = 8 , eta = 0.5 )


# In[107]:


# selecting the days in which rain has occured , so we can predict the rain amount 
rain_days = data[data['Rainy Days '] == 1]
rain_days.shape


# In[108]:


X = rain_days.drop(columns=['Rain'])
y = rain_days['Rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=rain_days['Rainy Days '])


# In[109]:


modelR.fit(X_train, y_train)
y_pred = modelR.predict(X_test)


# In[110]:


df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

# Select only the first 30 rows
df_first_30 = df.head(30)
# plot
plt.figure(figsize=(10, 6))

sns.lineplot(x=df_first_30.index, y='y_test', data=df_first_30, label='Actual Values (y_test)')
sns.lineplot(x=df_first_30.index, y='y_pred', data=df_first_30, label='Predicted Values (y_pred)')

plt.title('First 30 Actual vs Predicted Values')
plt.xlabel('Data Point Index')
plt.ylabel('Value')

plt.legend()
plt.show()


# In[111]:


plt.scatter(y_pred , y_test)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[112]:


# results time , here is the model performance

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)


print("MAE:", mae)
print("R2 Score:", r2)
print("RMSE:", rmse)

# This are the results archived
# MAE: 13.080879848480224
# R2 Score: 0.4296928636777182
# RMSE: 23.416952182514603


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




