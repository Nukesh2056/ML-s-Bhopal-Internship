#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning  

# In[1]:


# objective is to forecast the next day max temperature .

import pandas as pd 
data = pd.read_csv('combined_weather_data.csv')
data = data.drop('cloud cover' , axis=1)
# data.isna().sum()
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
data.shape


# In[6]:


data.columns


# In[7]:


data.isna().sum()


# In[8]:


data['Max.'] = data['Max.'].fillna(data['Max.'].median())
data['Morn'] = data['Morn'].fillna(data['Morn'].median())
data['Eve'] = data['Eve'].fillna(data['Eve'].median())
data['PE'] = data['PE'].fillna(data['PE'].median())


# In[9]:


data.isna().sum() # data is now cleaned 


# In[10]:


data.to_csv('combined & cleaned.csv')


# # Model Traning --> ARIMA

# In[11]:





# In[12]:


# ARIMA model is for only one feature i.e it can take only one feature, so we will use it here to predict the next day maximum temperature.
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima.model import ARIMA


# In[13]:


data = data.sort_index()  # Sort by date


# In[14]:


data


# In[15]:


# checking for any null values
data.isna().sum()


# In[16]:


columns_for_maxtemp = ['Date','Max.'] 
max_temp = data[columns_for_maxtemp]
max_temp


# In[17]:


# conveting the date column into date time format  & sorting dates
max_temp['Date'] = pd.to_datetime(max_temp['Date'] , format= '%d.%m.%y')
max_temp = max_temp.sort_values(by='Date' )


# In[18]:


plt.plot(max_temp['Date'],max_temp['Max.'])
plt.xticks(rotation=45)
plt.tight_layout()


# In[19]:


ts_data = max_temp[['Date', 'Max.']].copy()
ts_data.set_index('Date', inplace=True)


# In[20]:


ts_data.sort_index(inplace=True)

ts_data


# In[21]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf , plot_acf
from sklearn.metrics import mean_squared_error , mean_absolute_error , root_mean_squared_error


# In[22]:


result = adfuller(ts_data['Max.'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])


# In[23]:


# arima parameters ---> order(p , d , q )
# p : Autoregressive (AR) Component : The number of lagged observations (past values) included in the model to predict the current value.
# d : Differencing (Integration) : The number of times the time series is differenced to make it stationary (i.e., to remove trends and stabilize the mean)
# q : Moving Average (MA) Component : The MA component models the relationship between the current value and past forecast errors.
model = ARIMA(ts_data['Max.'].iloc[ : 3950 ], order=(2, 0, 3))  # d=0 since data is stationary
model_fit = model.fit()
print(model_fit.summary())


# In[24]:


residuals = model_fit.resid


# In[ ]:





# ## PACF plot for p parameter 

# In[25]:


# PACF plot -> PACF plot shows the corelation of a day from its respective lag days , it removes indirect corelation .
# lag dasy means past shift values.

plot_pacf(max_temp['Max.'], lags=20, method='ywm')
# this plot shown that we can set the value of p = 1 , setting the parameters will make the model better 

plt.title('Partial Autocorrelation Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.tight_layout()
plt.show()


# ## ACF Plot for q parameter

# In[26]:


# PACF plot -> PACF plot shows the corelation of a day from its respective lag days weather is direct or indirect.

plot_acf(ts_data['Max.'], lags=20)  # lags=30 means look 30 days back
plt.title('ACF Plot of Total_Amount')
plt.show()


# In[27]:


print(model_fit.summary())


# In[28]:


forecast = model_fit.forecast(steps=10)

mae = mean_absolute_error(ts_data[ 3950 : 3960], forecast)
rmse = np.sqrt(mean_squared_error(ts_data[ 3950 : 3960], forecast))

print(mae,rmse)


# # XGBoost Regressor

# In[29]:


from xgboost import XGBRegressor
import numpy as np
df = pd.read_csv('combined & cleaned.csv')


# In[30]:


df


# In[31]:


# Rename columns for easier handling
df.rename(columns={'Max.': 'Max', 'Min.': 'Min', 'Rainy Days ': 'Rainy_Days'}, inplace=True)


# In[32]:


cols_to_drop = ['Unnamed: 0', 'Year', 'Morn', 'Eve', 'PE']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)


# In[33]:


df['Date'] = pd.to_datetime(df['Date'])
df = df.groupby('Date').mean().reset_index()
df = df.sort_values('Date')


# In[34]:


# Drop rows with missing values in essential columns
df = df.dropna(subset=['Max', 'Min', 'Rain', 'Rainy_Days'])


# In[35]:


df['Next_Max'] = df['Max'].shift(-1)


# In[36]:


df['Next_Month'] = df['Date'].dt.month.shift(-1)
df['Next_DayOfYear'] = df['Date'].dt.dayofyear.shift(-1)

df['Max_lag1'] = df['Max'].shift(1)
df['Min_lag1'] = df['Min'].shift(1)
df['Rain_lag1'] = df['Rain'].shift(1)
df['Rainy_Days_lag1'] = df['Rainy_Days'].shift(1)


# In[37]:


df = df.dropna().reset_index(drop=True)


# In[38]:


df


# In[39]:


feature_cols = ['Max_lag1', 'Min_lag1', 'Rain_lag1', 'Rainy_Days_lag1', 'Next_Month', 'Next_DayOfYear']
X_train = df[feature_cols].iloc[ :3000 ]
y_train = df['Next_Max'].iloc[ :3000 ]

X_test = df[feature_cols].iloc[3000 : ]
y_test = df['Next_Max'].iloc[3000 :]


# In[40]:


model = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)


# In[41]:


# Prepare data for forecasting
last_row = df.iloc[-1]
last_date = last_row['Date']
last_max = last_row['Max']
last_min = last_row['Min']
last_rain = last_row['Rain']
last_rainy = last_row['Rainy_Days']


# In[42]:


# Forecast next 30 days
predictions = []
forecast_dates = []


# In[43]:


for i in range(1, 31):
    next_date = last_date + pd.Timedelta(days=i)
    next_month = next_date.month
    next_doy = next_date.dayofyear

    input_features = np.array([[last_max, last_min, last_rain, last_rainy, next_month, next_doy]])
    predicted_max = model.predict(input_features)[0]

    predictions.append(predicted_max)
    forecast_dates.append(next_date)

    # Update for next iteration (only updating max temp; keeping other vars same)
    last_max = predicted_max


# In[44]:


forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Predicted_Max_Temp': predictions
})


# In[45]:


print("\nForecasted Maximum Temperatures for Next 30 Days:")
print(forecast_df)


# In[46]:


y_predicted = model.predict(X_test)


# In[47]:


mae = mean_absolute_error(y_test, y_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
print("MAE:", mae)
print("RMSE:", rmse)


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





# In[ ]:





# In[ ]:





# In[ ]:




