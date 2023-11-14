#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:/Users/smita/Downloads/COVID-19 Coronavirus.csv")
df


# In[3]:


df.shape


# In[4]:


df.size


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


#check row which contains null values (Hint axis=1)
nan_df=df[df.isna().any(axis=1)]
nan_df.head()


# In[9]:


df.isna().sum()


# In[10]:


df.duplicated().value_counts()


# In[11]:


dfa = df.drop_duplicates(subset=None, keep='first', inplace=False)
dfa


# In[12]:


# fill null value with most suitable value(hint to replace, use implace-True df.fillna('Montenegro', implace=True))
df.fillna('Montenegro', inplace=True)
df


# In[13]:


df.loc[135]


# In[14]:


df.describe()


# In[15]:


top_5_countries=df.nlargest(n=5, columns="Total Cases")
top_5_countries


# In[17]:


#plot bargraph for top 5 countries
plt.figure(figsize=(10, 4), dpi=75)
sns.barplot(data=top_5_countries, x="Country", y="Total Cases")
plt.title("Top 5 Countries with Highest Total Cases")
plt.xlabel("Country")
plt.ylabel("Total Cases")


# In[18]:


plt.figure(figsize=(10, 4), dpi=75)
sns.barplot(data=top_5_countries, x="Country", y="Total Cases")
plt.title("Top 5 Countries with Highest Total Cases")
plt.xlabel("Country")
plt.ylabel("Total Cases")
plt.show()


# In[20]:


plt.figure(figsize=(10, 4), dpi=75)
sns.barplot(data=top_5_countries, x="Country", y="Total Deaths")
plt.title("Top 5 Countries with Highest Total Deaths")
plt.xlabel("Country")
plt.ylabel("Total Deaths")
plt.show()


# In[22]:


# top 5 countries which have more number of covid deaths percentage
top_5_death_per=df.nlargest(n=5, columns="Death percentage")
top_5_death_per


# In[24]:


plt.figure(figsize=(10, 4), dpi=75)
sns.barplot(data=top_5_death_per, x="Country", y="Total Deaths")
plt.title("Top 5 Countries with Highest Total Deaths")
plt.xlabel("Country")
plt.ylabel("Total Deaths")
plt.show()


# In[26]:


case_continent=df.groupby('Continent').sum()
case_continent


# In[28]:


case_continent = case_continent.reset_index()
plt.figure(figsize=(10,8), dpi=122)
sns.barplot(data=case_continent, x="Continent", y="Total Cases")
plt.xlabel("Continent")
plt.ylabel("Total Cases")
plt.show()


# In[29]:


plt.figure(figsize=(15, 10))
plt.xlabel('Population', fontsize=15)
plt.ylabel('Total Cases', fontsize=15)
plt.title('Scatter plot of Population Vs Total Cases', fontsize=15)
sns.scatterplot(data=df, x='Population', y='Total Cases', color='black')
plt.show()


# In[31]:


plt.figure(figsize=(15, 10))
plt.xlabel('Total Cases', fontsize=15)
plt.ylabel('Total Deaths', fontsize=15)
plt.title('Scatter plot of Total Cases Vs Total Deaths', fontsize=15)
sns.scatterplot(data=df, x='Total Cases', y='Total Deaths', color='black')
plt.show()


# In[33]:


df.corr(method="spearman")#selling the method as sperman
plt.figure(figsize=(20, 8)) #setting the figuresize

heatmap = sns.heatmap(df.corr(method='spearman').round(3), vmin=1, vmax=1, annot=True) #annot= True means writing the data value
font2 = {'family':'serif','color':'green','size':20}
plt.title("Spearman Rank Correlation", font2)
plt.show() #displaying heatmap


# In[ ]:




