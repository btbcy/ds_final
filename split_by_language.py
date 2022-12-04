#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('train_2.csv')


# In[39]:


df


# In[3]:


df.info()


# In[67]:


df_en = df.loc[df['Page'].str.contains('_en.wiki')]
# df_en.info()

en_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    en_total.iloc[0][col] = df_en[col].sum()
en_total = en_total.transpose()
# en_total.plot(figsize=(15,10))

analy = df_en.iloc[:, 380:440]
s = analy.sum(axis=1)
s = s.sort_values(ascending = False)[0:30].index.tolist()
for i in s:
    print(df['Page'][i])


# In[18]:


df_zh = df.loc[df['Page'].str.contains('_zh.wiki')]
df_zh.info()
zh_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    zh_total.iloc[0][col] = df_zh[col].sum()
zh_total = zh_total.transpose()
zh_total.plot(figsize=(15,10))


# In[19]:


df_es = df.loc[df['Page'].str.contains('_es.wiki')]
df_es.info()
es_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    es_total.iloc[0][col] = df_es[col].sum()
es_total = es_total.transpose()
es_total.plot(figsize=(15,10))


# In[9]:


df_ru = df.loc[df['Page'].str.contains('_ru.wiki')]
# df_ru.info()
ru_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    ru_total.iloc[0][col] = df_ru[col].sum()
ru_total = ru_total.transpose()
# ru_total.plot(figsize=(15,10))

# print(ru_total[385:415])
ru_peak = df_ru.iloc[:, 385:415]
ru_peak_col = ru_peak.sum(axis=1)
ru_peak_col = ru_peak_col.sort_values(ascending = False)[0:15].index.tolist()
# for i in ru_peak_col:
#     print(i)
#     print(df['Page'][i])
test = df.iloc[100890]
test[1:].plot(figsize=(15,10))


# In[21]:


df_de = df.loc[df['Page'].str.contains('_de.wiki')]
df_de.info()
de_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    de_total.iloc[0][col] = df_de[col].sum()
de_total = de_total.transpose()
de_total.plot(figsize=(15,10))


# In[22]:


df_ja = df.loc[df['Page'].str.contains('_ja.wiki')]
df_ja.info()
ja_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    ja_total.iloc[0][col] = df_ja[col].sum()
ja_total = ja_total.transpose()
ja_total.plot(figsize=(15,10))


# In[23]:


df_fr = df.loc[df['Page'].str.contains('_fr.wiki')]
df_fr.info()
fr_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    fr_total.iloc[0][col] = df_fr[col].sum()
fr_total = fr_total.transpose()
fr_total.plot(figsize=(15,10))


# In[ ]:


df_de = df.loc[df['Page'].str.contains('_de.wiki')]
df_de.info()
de_total = pd.DataFrame(columns=df.columns[1:], index=['sum'])
for col in df.columns[1:]:
    de_total.iloc[0][col] = df_de[col].sum()
de_total = de_total.transpose()
de_total.plot(figsize=(15,10))

