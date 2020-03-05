#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns
import plotly.graph_objs as go 
import plotly.offline as py
color = sns.color_palette()
#py.init_notenotebook_mode(connected=True)
import plotly.tools as tlsdata 
import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from gensim.models import word2vec

from sklearn.manifold import TSNE
from sklearn import metrics
import sklearn
from sklearn.metrics import jaccard_similarity_score
cv = CountVectorizer()
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
stop = set(stopwords.words("english"))

import warnings 
warnings.filterwarnings('ignore')
import os
os.listdir("../input")

#input data files are available in the "../input" directory
#for example, running this will list the files in the input directory
data = pd.read_csv('../input/1429_1.csv', encoding="ISO-8859-1")
#keeping only the neceessary columns
#print(data.head())

#any results you write to the current directory are saved as output.

#----------------------------------------------------

print(data.shape)
print(data.dtypes)
print(data.isnull().sum())
data = data.dropna(subset=['reviews.text'])

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1
        ).generate(str(data))

    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplot_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


show_wordcloud(data['reviews.text'])


# In[2]:


cnt_srs = data['reviews.rating'].value_counts().head()
trace = go.Bar(
    y=cnt_srs.index[::-1], 
    x=cnt_srs.values[::-1], 
    orientation='h', 
    marker=dict(
        color=cnt_srs.values[::-1], 
        colorscale='Blues', 
        reversescale=True), 
    )
layout = dict(title='Ratings distribution')
data1 = [trace]
fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename='Ratings')


# In[3]:


from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from IPython.display import HTML
cat_hist = data.groupby('categories', as_index=False).count()
HTML(pd.DataFrame(cat_hist['categories']).to_html())
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
def removePunctuation(x):
    x = x.lower()
    x = re.sub(r'[^\x00-\x7f]',r' ', x) # replace the char that is not in the ASCII table
    return re.sub('['+string.punctuation+']', " ", x)

stops = set(stopwords.words("english"))
def removeStopwords(x):
    filterd_words = [word for word in x.split() if word not in stops]
    return "".join (filtered_words)
'''
When we deal with text problem in Natural Language Processing, 
stop words removal process is a one of the important step to have a 
better input for any models. Stop words means that it is a very 
common words in a language (e.g. a, an, the in English. 的, 了 in 
Chinese. え, も in Japanese). It does not help on most of NLP problem 
such as semantic analysis, classification etc.
'''

def removeAmzString(x):
    return re.sub(r'[0-9]+ people found this helpful\. Was this review helpful to you Yes No', "", x)
# remove the amazon fixed sentence. 


# In[4]:


reviews =  [sent if type(sent)==str else "" for sent in data['reviews.title'].values]
reviews = [removeAmzString(sent) for sent in reviews]

reviews = [removePunctuation(sent) for sent in reviews]

stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=200,
                      max_font_size=40,random_state=42).generate(str(reviews))
plt.figure(figsize=(15,20))
ax1 = plt.subplot2grid((4, 2), (0, 0))
ax2 = plt.subplot2grid((4, 2), (1, 0))
ax3 = plt.subplot2grid((4, 2), (0, 1), rowspan=2)
ax4 = plt.subplot2grid((4, 2), (2, 0), colspan=2,rowspan=2)

rat_hist = data.groupby('reviews.rating',as_index=False).count()
sns.barplot(x=rat_hist['reviews.rating'].values,y=rat_hist['id'].values,ax=ax1)

cat_hist = cat_hist.sort_values(by='id')
sns.barplot(x=cat_hist['categories'].index,y=cat_hist['id'].values,ax=ax3)

hf_hist = data.groupby('reviews.numHelpful',as_index=False).count()[0:30]
sns.barplot(x=hf_hist['reviews.numHelpful'].values.astype(int),y=hf_hist['id'].values,ax=ax2)

ax1.set_title("Reviews Ratings",fontsize=16)
ax3.set_title("Categories",fontsize=16)
ax2.set_title("Helpful Feedback",fontsize=16)
ax4.set_title("Words Cloud",fontsize=16)
ax4.imshow(wordcloud)
ax4.axis('off')
plt.show()


# In[36]:


import nltk.stem as ns
from spellchecker import SpellChecker
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s = s.replace(' s ', ' is ')
    s = s.replace('don t', 'do not')
    s = s.replace('doesn t', 'does not')
    s = s.replace('can t', 'cannot')
    s = s.replace('isn t', 'is not')
    s = s.replace('uldn t', 'uld not')
    s = s.replace('aren t', 'are not')
    s = s.replace('wasn t', 'was not')
    s = s.replace('weren t ', 'were not')
    s = s.replace('haven t', 'have not')
    s = s.replace('hasn t', 'has not')
    s = s.replace(' ve ', ' have ')
    s = s.replace(' wa ', ' was ')
    s = s.replace(' s ','')
    s = s.replace(' t ','')
    s = s.replace(' ntact', ' contact')
    s = s.replace(' nnect', ' connect')
    s = s.replace(' wasnt ', ' was not ')
    words = s.split()
    lemmatizer = ns.WordNetLemmatizer()
    for i in range(len(words)):
        words[i] = lemmatizer.lemmatize(words[i],'n')
    s = " ".join(words)
    return s
data['reviews.text'] = [cleaning(s) for s in data['reviews.text']]
data['revies.title'] = [cleaning(s) for s in data['reviews.title']]


# In[37]:


from pathlib import Path
data = data.reset_index(drop=True)
outfile = Path('reviews.txt')
with outfile.open('w',encoding='utf-8') as w:
    for k in range(data['reviews.text'].shape[0]):
        w.write(data['reviews.text'][k]+'\n')


# In[38]:


from sklearn.model_selection import train_test_split
from collections import  Counter
review = data['reviews.text'].to_numpy()
rate = data['reviews.rating'].to_numpy()
idx = np.argwhere(np.isnan(rate))
review_miss_rate = review[idx]
rate_miss = rate[idx]
review1 = np.delete(review, idx.T, axis=0)
rate1 = np.delete(rate, idx.T, axis=0)
review_train, review_test, rate_train, rate_test = train_test_split(review1,
                        rate1,test_size=0.2,random_state=10,stratify=rate1)
pd.DataFrame({'review':review_train}).to_csv('train_review.csv')
pd.DataFrame({'review':review_test}).to_csv('test_review.csv')
pd.DataFrame({'rate':rate_train}).to_csv('train_rate.csv')
pd.DataFrame({'rate':rate_test}).to_csv('test_rate.csv')
pd.DataFrame({'review':review_miss_rate.reshape(33)}).to_csv('miss_review_rate.csv')
pd.DataFrame({'rate':rate_miss.reshape(33)}).to_csv('miss_rate.csv')


# In[39]:


outfile = Path('train_review.txt')
with outfile.open('w',encoding='utf-8') as w:
    for k in range(review_train.shape[0]):
        w.write(review_train[k]+'\n')

outfile = Path('test_review.txt')
with outfile.open('w',encoding='utf-8') as w:
    for k in range(review_test.shape[0]):
        w.write(review_test[k]+'\n')


# In[ ]:


from sklearn.model_selection import train_test_split
df_train = pd.read_csv('review_train.csv')
df_train = df_train.dropna(axis=0,how='any')
df_train = df_train.reset_index(drop=True)
x = df_train['review'].to_numpy()
y = pd.read_csv('rate_train.csv')
y = y['rate'].to_numpy()
y = np.delete(y,12675,axis=0)# we don't have comment on idx = 12675
# whole training set
#train, val, y_train, y_val = train_test_split(x,
#                        y,test_size=0.2,random_state=10,stratify=y)


# In[ ]:


from pathlib import Path
outfile = Path('../input/train.txt')
with outfile.open('w',encoding='utf-8') as w:
    for k in range(train.shape[0]):
        w.write(train[k]+'\n')
pd.DataFrame({'review':train}).to_csv('../input/train.csv')


# In[60]:


from pathlib import Path
outfile = Path('../input/val.txt')
with outfile.open('w',encoding='utf-8') as w:
    for k in range(val.shape[0]):
        w.write(val[k]+'\n')
pd.DataFrame({'review':val}).to_csv('../input/val.csv')


# In[ ]:


pd.DataFrame({'rate':y_train}).to_csv('../input/y_train.csv')
pd.DataFrame({'rate':y_val}).to_csv('../input/y_val.csv')

