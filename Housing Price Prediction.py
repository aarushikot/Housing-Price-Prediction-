#!/usr/bin/env python
# coding: utf-8

# # Importing Necessaries libraries

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings as ww
ww.filterwarnings('ignore')


# # Loading the dataset

# In[2]:


## Data taken from kaggle


# In[3]:


df = pd.read_csv("housing_price_dataset.csv.zip")


# # Data Preprocessing

# In[4]:


df.head()


# In[5]:


df.info()


# In[7]:


# Checking for null values


# In[6]:


df.isnull().sum()


# In[8]:


# Checking for mean, median, std, min, max....


# In[9]:


df.describe()


# In[ ]:





# In[17]:


sns.set_style('whitegrid')
df[['SquareFeet','Bedrooms','Bathrooms','Price']].hist(grid=False)
plt.tight_layout()


# In[ ]:


# Count plot for categorical variable


# In[20]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Neighborhood', data=df)
plt.title('Count of Neighborhood')
plt.show()


# In[ ]:


# Scatter plot for two numerical variables (Price and SquareFeet)


# In[22]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='SquareFeet', y='Price', data=df)
plt.title('Scatter Plot between SquareFeet and Price')
plt.show()


# In[23]:


# Scatter plot for two numerical variables (Bedrooms and Price)


# In[26]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Bedrooms', y='Price', data=df)
plt.title('Scatter Plot between Bedrooms and Price')
plt.show()


# In[27]:


# Scatter plot for two numerical variables (Year Built and Price)


# In[29]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearBuilt', y='Price', data=df)
plt.title('Scatter Plot between YearBuilt and Price')
plt.show()


# In[ ]:





# In[ ]:





# In[30]:


# Box plot for numerical variable vs. categorical variable
plt.figure(figsize=(10, 6))
sns.boxplot(x='Neighborhood', y='Price', data=df)
plt.title('Box Plot of Price by Neighborhood ')
plt.show()


# In[18]:


# Correlation matrix


# In[19]:


correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[21]:


sns.histplot(df['Price'], kde=True)


# # The dataset is pretty balanced and the price follows a normal distribution
# # Now we transform the non-numerical data to numerical data using dummy variable

# In[31]:


df = pd.get_dummies(data=df, columns=['Neighborhood'])
df.replace({True:1, False:0}, inplace=True)
df.head()


# In[32]:


# Show the correlation matrix


# In[33]:


correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# # We can see clearly above the independant relations between the variables and the Price variable
# 
# # Remove the YearBuilt feature
# 

# In[34]:


df.drop(columns='YearBuilt', inplace=True)
df.head()


# In[ ]:





# In[36]:


X = df.drop(columns='Price')
Y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.85)


# In[37]:


model = LinearRegression()


# In[39]:


model.fit(X_train, y_train)


# In[48]:


# Training a linear regression model to see how it behaves

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import median_absolute_error

lin_model = LinearRegression()
poly2_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly3_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

lin_model.fit(X_train, y_train)
poly2_model.fit(X_train, y_train)
poly3_model.fit(X_train, y_train)

scores = pd.DataFrame({
    'MAE': {
        'Linear Model': median_absolute_error(y_test, lin_model.predict(X_test)),
        'Polynomial Model (degree = 2)': median_absolute_error(y_test, poly2_model.predict(X_test)),
        'Polynomial Model (degree = 3)': median_absolute_error(y_test, poly3_model.predict(X_test))
    },
    'R2_score': {
        'Linear Model': r2_score(y_test, lin_model.predict(X_test)),
        'Polynomial Model (degree = 2)': r2_score(y_test, poly2_model.predict(X_test)),
        'Polynomial Model (degree = 3)': r2_score(y_test, poly3_model.predict(X_test))
    }
})

scores


# In[42]:


pip install xgboost


# In[53]:


# For the linear model we supposed that the variables are quite independant
# We'll use now models that omit this supposition

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
params = {
    'n_estimators': [100, 2500],
    'learning_rate': [0.1, 0.01],
    'max_depth': [2,6]
}

xgb_model_opt = GridSearchCV(xgb_model, params, scoring='neg_median_absolute_error', verbose=3)
xgb_model_opt.fit(X_train, y_train)

xgb_model_opt.best_params_, xgb_model_opt.best_score_


# In[54]:


# Evaluating the models
models = {
    'Linear Model': lin_model,
    'Polynomial Model (degree = 2)': poly2_model,
    'Polynomial Model (degree = 3)': poly3_model,
    'XGBoost Model' : xgb_model_opt.best_estimator_
}

tmp_scores = {}
for m in list(models.keys()):
    tmp_scores[m] = median_absolute_error(y_test, models[m].predict(X_test))
    

scores = pd.DataFrame({
    'MAE': tmp_scores

    
})

scores


# In[59]:


new_data = pd.DataFrame({'SquareFeet': [2000], 'Bedrooms': [3],'Bathrooms':[2],'Neighborhood_Rural': [1],'Neighborhood_Suburb':[0],'Neighborhood_Urban':[0]})
predicted_price = xgb_model_opt.predict(new_data)
print(f'Predicted Price for New Data: {predicted_price[0]}')


# In[ ]:




