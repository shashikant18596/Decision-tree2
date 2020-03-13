#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd 
from sklearn.tree import DecisionTreeRegressor


# In[2]:


df = pd.read_csv('C:\\Users\shashikant\Desktop\Decision Tree\decision2.csv')
df


# In[13]:


dummy_company = pd.get_dummies(df.company)
dummy_company


# In[10]:


dummy_job = pd.get_dummies(df.job)
dummy_job


# In[11]:


dummy_degree = pd.get_dummies(df.degree)
dummy_degree


# In[16]:


df1 = pd.concat([df,dummy_company],axis = 'columns')
df1


# In[17]:


df2 = pd.concat([df1,dummy_job],axis = 'columns')
df2


# In[18]:


df3 = pd.concat([df2,dummy_degree],axis = 'columns')
df3


# In[20]:


new_df = df3
new_df


# In[31]:


x = new_df.drop(['salary','company','job','degree'],axis='columns').values
x


# In[32]:


y = new_df[['salary']].values
y


# In[35]:


new_df


# In[30]:


model = DecisionTreeRegressor()


# In[33]:


model.fit(x,y)


# In[34]:


model.score(x,y)


# In[38]:


model.predict([[0, 1, 0, 0, 1, 0, 0, 1]])


# In[ ]:




