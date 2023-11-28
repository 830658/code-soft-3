#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
for dirname. _. filenamaes in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# In[3]:


df=pd.read_csv('IRIS.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


#visualising different columns wrt species

plt.figure(figsize=(16, 6))
sns.countplot(x='sepal_length', data=df, palette='inferno', hue='species')

plt.show()


# In[7]:


plt.figure(figsize=(16, 6))
sns.countplot(x='sepal_width', data=df, palette='Reds', hue='species')

plt.show()


# In[8]:


plt.figure(figsize=(16, 6))
sns.countplot(x='petal_length', data=df, palette='magma', hue='species')

plt.show()


# In[9]:


plt.figure(figsize=(16, 6))
sns.countplot(x='petal_width', data=df, palette='Blues', hue='species')

plt.show()


# In[10]:


plt.figure(figsize= (8,8))
sns.scatterplot(x='sepal_length',y='sepal_width',data = df ,hue = 'species', palette= 'inferno',s=60)


# In[11]:


plt.figure(figsize= (8,8))
sns.scatterplot(x='petal_length',y='petal_width',data = df ,hue = 'species', palette= 'inferno',s=60)


# In[12]:


plt.figure(figsize = (8,8))
sns.pairplot(df)


# In[13]:


#LOGISTIC REGRESSION MODEL
x=df.drop(['species'],axis=1)
y=df['species']


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)


# In[15]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression


# In[16]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming df is your DataFrame containing the Iris dataset
X = df.drop('species', axis=1)
y = df['species']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = lr.predict(X_train)

# Evaluate the model on the training set
train_score = r2_score(y_train, y_train_pred)
print("Training R2 Score:", train_score)


# In[17]:


predictions= lr.predict(x_test)
predictions


# In[18]:


y_test


# In[23]:


# Assuming X_train, X_test, y_train, y_test are your training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

