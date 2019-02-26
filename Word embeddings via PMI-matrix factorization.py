
# coding: utf-8

# # Word Embeddings via PMI Matrix Factorization
# 
# *Contact TA: emaad[at]cmu.edu, [eyeshalfclosed.com/teaching/](http://www.eyeshalfclosed.com/teaching/)*
# 
#    * Based on [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization), by Omar Levy and Yoav Goldberg, NIPS 2014.
#    * Dataset: https://www.kaggle.com/hacker-news/hacker-news-posts/downloads/HN_posts_year_to_Sep_26_2016.csv
#    * Notes: http://www.eyeshalfclosed.com/teaching/95865-recitation-word2vec_as_PMI.pdf
#    * Source material: https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/.
#    * Source material: https://www.kaggle.com/alexklibisz/simple-word-vectors-with-co-occurrence-pmi-and-svd

# In[37]:


from collections import Counter
from itertools import combinations
from math import log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pformat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, norm
from string import punctuation


# ## 0. Load the data.

# In[6]:


df = pd.read_csv('HN_posts_year_to_Sep_26_2016.csv', usecols=['title'])
df.head()


# ## 1. Read and preprocess titles from HN posts.

# In[10]:


get_ipython().run_cell_magic('time', '', "punctrans = str.maketrans(dict.fromkeys(punctuation))\ndef tokenize(title):\n    x = title.lower() # Lowercase\n    x = x.encode('ascii', 'ignore').decode() # Keep only ascii chars.\n    x = x.translate(punctrans) # Remove punctuation\n    return x.split() # Return tokenized.\n\ntexts_tokenized = df['title'].apply(tokenize)")


# ## 2a. Compute unigram and bigram counts.
# 
# A unigram is a single word (x). A bigram is a pair of words (x,y).
# Bigrams are counted for any two terms occurring in the same title.
# For example, the title "Foo bar baz" has unigrams [foo, bar, baz]
# and bigrams [(bar, foo), (bar, baz), (baz, foo)]

# In[11]:


get_ipython().run_cell_magic('time', '', 'cx = Counter()\ncxy = Counter()\nfor text in texts_tokenized:\n    for x in text:\n        cx[x] += 1\n    for x, y in map(sorted, combinations(text, 2)):\n        cxy[(x, y)] += 1')


# ## 2b. Remove frequent and infrequent unigrams.
# 
# Pick arbitrary occurrence count thresholds to eliminate unigrams occurring
# very frequently or infrequently. This decreases the vocab size substantially.

# In[13]:


get_ipython().run_cell_magic('time', '', "print('%d tokens before' % len(cx))\nmin_count = (1 / 1000) * len(df)\nmax_count = (1 / 50) * len(df)\nfor x in list(cx.keys()):\n    if cx[x] < min_count or cx[x] > max_count:\n        del cx[x]\nprint('%d tokens after' % len(cx))\nprint('Most common:', cx.most_common()[:25])")


# ## 2c. Remove frequent and infrequent bigrams.
# 
# Any bigram containing a unigram that was removed must now be removed.

# In[15]:


get_ipython().run_cell_magic('time', '', 'for x, y in list(cxy.keys()):\n    if x not in cx or y not in cx:\n        del cxy[(x, y)]')


# ## 3. Build unigram <-> index lookup.

# In[16]:


get_ipython().run_cell_magic('time', '', 'x2i, i2x = {}, {}\nfor i, x in enumerate(cx.keys()):\n    x2i[x] = i\n    i2x[i] = x')


# ## 4. Sum unigram and bigram counts for computing probabilities.
# 

# In[17]:


sx = sum(cx.values())
sxy = sum(cxy.values())


# # 5. Accumulate data, rows, and cols to build sparse PMI matrix
# 
# The PMI value for a bigram with tokens (x, y) is:
# $$ \textrm{PMI}(x,y) = \frac{\textrm{log}(p(x,y))}{p(x)p(y)} $$
# 
# The probabilities are computed on the fly using the sums from above.

# In[19]:


get_ipython().run_cell_magic('time', '', "pmi_samples = Counter()\ndata, rows, cols = [], [], []\nfor (x, y), n in cxy.items():\n    rows.append(x2i[x])\n    cols.append(x2i[y])\n    data.append(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))\n    pmi_samples[(x, y)] = data[-1]\nPMI = csc_matrix((data, (rows, cols)))\nprint('%d non-zero elements' % PMI.count_nonzero())\nprint('Sample PMI values\\n', pformat(pmi_samples.most_common()[:10]))")


# ## 6. Factorize the PMI matrix using sparse SVD aka "learn the unigram/word vectors".
# 
# This part replaces the stochastic gradient descent used by Word2vec
# and other related neural network formulations. We pick an arbitrary vector size k=20.

# In[21]:


get_ipython().run_cell_magic('time', '', 'U, _, _ = svds(PMI, k=20)')


# Normalize the vectors to compute cosine similarity.

# In[22]:


norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
U /= np.maximum(norms, 1e-7)


# ## 8. Show some nearest neighbor samples as a sanity-check.
# 
# The format is `<unigram> <count>: (<neighbor unigram>, <similarity>), ...`
#     
# From this we can see that the relationships make sense.

# In[25]:


k = 5
for x in ['facebook', 'twitter', 'instagram', 'messenger', 'hack', 'security', 
          'deep', 'encryption', 'cli', 'venture', 'paris']:
    dd = np.dot(U, U[x2i[x]]) # Cosine similarity for this unigram against all others.
    s = ''
    # Compile the list of nearest neighbor descriptions.
    # Argpartition is faster than argsort and meets our needs.
    for i in np.argpartition(-1 * dd, k + 1)[:k + 1]:
        if i2x[i] == x: continue
        s += '(%s, %.3lf) ' % (i2x[i], dd[i])
    print('%s, %d\n %s' % (x, cx[x], s))
    print('-' * 10)


# ## 9. Word-vector compositions

# In[63]:


composition = U[x2i["facebook"]] - U[x2i["ads"]]
composition /= np.linalg.norm(composition)

k = 2
composition = U[x2i["facebook"]] + U[x2i["images"]]
dd = np.dot(U, composition) # Cosine similarity for this unigram against all others.
s = ''
for i in np.argpartition(-1 * dd, k + 1)[:k + 1]:
    s += '(%s, %.3lf) ' % (i2x[i], dd[i])
print(s)

