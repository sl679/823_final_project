#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings
warnings.filterwarnings('ignore')
from pandas import DatetimeIndex
from pandas import Timestamp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn import svm
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import plotly.offline as py 
import plotly.graph_objects as go
import plotly.express as px
pd.set_option('display.float_format', lambda x: '%.6f' % x)

import wget


# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[6]:




covid = pd.read_csv('/Users/shenjiajie/Desktop/Duke/823/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')

covid['submission_date'] = pd.to_datetime(covid['submission_date'])

covid = covid.sort_values(by = 'submission_date')


# In[7]:


state_data=covid[covid["state"]=="NC"]



start_date = '2020-12-14'
end_date = '2021-11-19'

mask = (state_data['submission_date'] >= start_date) & (state_data['submission_date'] <= end_date)
state_data = state_data.loc[mask]


datewise_state=state_data.groupby(["submission_date"]).agg({"tot_cases":'sum'})


# In[8]:


datewise_state["Days Since"]=datewise_state.index-datewise_state.index[0]
datewise_state["Days Since"]=datewise_state["Days Since"].dt.days

day=8
poly_degree_pr=3
svm_degree=5
poly_degree_polymlp=3

train_ml= datewise_state.iloc[:int(datewise_state.shape[0]*0.85)]

valid_ml=datewise_state.iloc[int(datewise_state.shape[0]*0.85):]


# In[9]:



def plot_result(data_d,new_date_time_index, forecast, gcolor, gcase, gtitle):
    
    plt.style.use('seaborn-white')
    plt.plot(data_d,label="Actual "+gcase,color=gcolor, linestyle='solid', linewidth = 3, marker='o', markerfacecolor=gcolor, markersize=12)
    plt.plot(new_date_time_index,forecast,label="Predicted "+gcase,color='black', linestyle='solid', linewidth = 3, marker='*', markerfacecolor='black', markersize=12)


# In[10]:


def new_forecast(prediction,new_prediction,new_date):
    
    forecast=np.concatenate((prediction,new_prediction))

    new_ar = []

    for single_timestamp in datewise_state.index:
        new_ar.append(pd.to_datetime(single_timestamp))

    for single_timestamp in new_date:
        new_ar.append(pd.to_datetime(single_timestamp))

    new_date_time_index = DatetimeIndex(new_ar, dtype='datetime64[ns]', name='Date', freq=None)

    return (forecast,new_date_time_index)




# In[12]:


def eval_reg(y,y_p):

    print("Mean Absolute Error: ",mean_absolute_error(y,y_p))

    print("R2-Squared:", r2_score(y,y_p))


# In[16]:



def MLPRegression_covid_19(train_d,valid_d,data_d,case):

    model_scores=[]

    
    mlp=MLPRegressor(hidden_layer_sizes=[20,10,5], max_iter=50000, alpha=0.0005, random_state=26)  
 
    mlp.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_d).reshape(-1,1))

    print(case)

    prediction_valid_mlp=mlp.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

    
    model_scores.append(np.sqrt(mean_squared_error(valid_d,prediction_valid_mlp)))
    print("RMSE for MLP: ",model_scores)

    eval_reg(valid_d,prediction_valid_mlp)
    
    prediction_mlp=mlp.predict(np.array(datewise_state["Days Since"]).reshape(-1,1))
 
  

    new_date=[]
    new_prediction_mlp=[]

    for i in range(1,day):
        new_date.append(datewise_state.index[-1]+timedelta(days=i))
        new_prediction_mlp.append(mlp.predict(np.array(datewise_state["Days Since"].max()+i).reshape(-1,1))[0]) 

    forecast_mlp, new_date_time_index =new_forecast(prediction_mlp,new_prediction_mlp,new_date)
    


    return (forecast_mlp, new_date_time_index, model_scores)


# In[25]:


t41 = time.process_time()

forecast_mlp_d, new_date_time_index_d, model_score_mlp_d = MLPRegression_covid_19(train_ml["tot_cases"],valid_ml["tot_cases"],datewise_state["tot_cases"],'Total Cases')


# In[24]:


fig1 = go.Figure()
 
    
fig1 = px.line(datewise_state, x=new_date_time_index_d, y=forecast_mlp_d )
fig1.add_scatter(x=new_date_time_index_d, y=datewise_state['tot_cases'])

  
fig1.update_traces(showlegend=True)

    
fig1.update_layout(
 title="The Time seris plot that shows the total cases of COVID-19 in North Carolina",
 xaxis_title="Time start from 2020-12-14 ",
 yaxis_title="Total number of COVID-19 cases ",
 legend_title="Legend Title: Purple(Predicted), Red(Actual Cases)",
 
     
 font=dict(
    family="Courier New, monospace",
    size=12,
    color="RebeccaPurple"
)
     

     
)
fig1.show()

