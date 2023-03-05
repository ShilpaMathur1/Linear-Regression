#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[7]:


advertising = pd.read_csv("C:/shilpa/upgrad/advertising.csv")
advertising.head()


# In[8]:


advertising.shape


# In[9]:


advertising.info()


# In[10]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[11]:


sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales',size=4)
plt.show()


# In[13]:


X = advertising['TV']
y = advertising['Sales']


# In[17]:


from sklearn.model_selection import train_test_split
X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[18]:


X_train_lm.shape


# In[21]:


X_train_lm = X_train_lm.values.reshape(-1,1)
X_test_lm = X_test_lm.values.reshape(-1,1)


# In[22]:


from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()

# Fit the model using lr.fit()
lm.fit(X_train_lm, y_train_lm)


# In[23]:


print(lm.intercept_)
print(lm.coef_)


# In[24]:


plt.scatter(X_train_lm, y_train_lm)
plt.plot(X_train_lm, 6.948 + 0.054*X_train_lm, 'r')
plt.show()


# In[25]:


from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()

# Fit the model using lr.fit()
lm.fit(X_test_lm, y_test_lm)


# In[26]:


print(lm.intercept_)
print(lm.coef_)


# In[27]:


plt.scatter(X_test_lm, y_test_lm)
plt.plot(X_test_lm, 6.726 + 0.059*X_test_lm, 'r')
plt.show()


# In[ ]:




