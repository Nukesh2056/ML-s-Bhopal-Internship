#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd 
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


# In[2]:


weather = pd.read_csv('ready_to_use_data_for_regression.csv')
weather.keys()


# In[3]:


df = pd.DataFrame(weather)
df


# In[4]:


x = df.drop(columns=['Next_Min' , 'Next_Max'])
y = df[['Next_Min', 'Next_Max']]

x_train , x_test , y_train , y_test = train_test_split(x , y , train_size=0.8 , random_state=42)


# In[5]:


model = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=100))
model.fit(x_train , y_train)


# In[6]:


y_predicted = model.predict(x_test)


# In[7]:


# Plot only first 100 points
n = 100

plt.figure(figsize=(12, 5))

# Min Temp
plt.subplot(1, 2, 1)
plt.plot(y_test['Next_Min'].values[:n], label='Actual Min', marker='o')
plt.plot(y_predicted[:n, 0], label='Predicted Min', marker='x')
plt.title('Min Temp (First 100 samples)')
plt.xlabel('Sample')
plt.ylabel('Temp')
plt.legend()

# Max Temp
plt.subplot(1, 2, 2)
plt.plot(y_test['Next_Max'].values[:n], label='Actual Max', marker='o')
plt.plot(y_predicted[:n, 1], label='Predicted Max', marker='x')
plt.title('Max Temp (First 100 samples)')
plt.xlabel('Sample')
plt.ylabel('Temp')
plt.legend()

plt.tight_layout()
plt.show()


# In[8]:


min_error = y_test['Next_Min'].values - y_predicted[:, 0]
max_error = y_test['Next_Max'].values - y_predicted[:, 1]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(min_error, bins=30, kde=True, color='green')
plt.title('Min Temp Error Distribution')

plt.subplot(1, 2, 2)
sns.histplot(max_error, bins=30, kde=True, color='blue')
plt.title('Max Temp Error Distribution')

plt.tight_layout()
plt.show()


# In[13]:


mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predicted)

print("Results :")
print(f"MAE: {mae:.2f} °C")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f} °C")
print(f"R² Score: {r2:.2f}")

