#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


animes = pd.read_csv('/Users/angelabi/SAAS RP/Anime csvs/animes.csv')
animes


# In[3]:


animes = animes.drop(labels=['synopsis','genre','episodes','members','img_url','link'], axis=1)
# dropped columns I wasn't interested in


# In[4]:


animes = animes.dropna(axis=0, how="any", thresh=None, subset=['score','popularity','ranked'], inplace=False)
animes
# dropped rows with nan values in score, popularity, ranked; 18732 rows -> 16099 rows
# 85.9% of original list of anime remains


# In[5]:


animes.describe()
# describe was most useful for looking at score


# In[6]:


sns.boxplot(animes['score'])
#visualising score distribution


# In[7]:


sns.histplot(data=animes, x="score")


# In[8]:


plt.figure(figsize=(20,10))
cor = animes.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()
#heatmap to look at correlation; looks like rank and popularity have high correlation of 0.85, which makes sense


# In[ ]:




