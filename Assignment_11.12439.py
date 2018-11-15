
# coding: utf-8

# Build the linear regression model using scikit learn in boston data to predict 'Price'
# based on other dependent variable.
# Here is the code to load the data

# In[1]:


#Import third party libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


# In[2]:


#import sklearn modules used for linear regresion
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# In[3]:


#Load boston housing pricing dataset
boston = load_boston()


# In[4]:


#boston data keys
boston.keys()


# In[5]:


#View boston dataset features
boston.feature_names


# In[6]:


#print(boston.DESCR)


# In[7]:


#View first 5 target values of boston dataset
boston.target[:5]


# In[8]:


#View features values of first 5 row in boston dataset
boston.data[:5]


# In[9]:


#Create DataFrame df_X, df_Y using boston dataset which contains features and target values
df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = boston.target


# In[10]:


#Statistical analysis of features of DataFrame df_X
df_x.describe()


# In[11]:


#Apply sklearn linear regression model
lr = LinearRegression()
lr


# In[12]:


#Split boston dataset into trainning and test dataset
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=5)


# In[13]:


#Fit linear regression model on training dataset
lr.fit(x_train,y_train)


# In[14]:


#check weight of all the features used in linear regression model
lr.coef_


# In[15]:


#estimated intercepts
lr.intercept_


# In[16]:


#Perform prediction on test dataset
pred = lr.predict(x_test)


# In[17]:


#first 10 predicted values calculated using the boston housing dataset features
pred[:10]


# In[18]:


#first 10 actual target values
y_test[:10]


# In[19]:


#Plot a figure for linear regression model
plt.xlabel("Actual Price ( $1000 )")
plt.ylabel("Predicted Price ( $1000 )")
plt.title("Actual vs Predicted Price")
plt.scatter(y_test, pred, color='green')
plt.show()


# In[20]:


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, pred))


# Ideally, the scatter plot should create a linear line. Since the model does not fit 100%, the scatter plot is not creating a linear line.

# That means that the model isn’t a really great linear model.

# In[21]:


#statistical analysis of linear regression model using statsmodels library
import statsmodels.formula.api as smf
df = df_x.copy()
df['target'] = boston.target
lm = smf.ols(formula='target ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=df).fit()
#lm.conf_int()
lm.summary()

