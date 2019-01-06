# -*- coding: utf-8 -*-
# imports needed and logging
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gzip
import gensim
import logging
import json
import os
import random
import numpy as np
import math
import torch
import random
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from torch.nn import init
from torch import optim
import time
from tqdm import tqdm_notebook as tqdm
import itertools


def sent_shuffle(sent):
    all_sents = []
    total_length = len(sent)

    for i in range(10):
        random.shuffle(sent)
        all_sents.append(sent[:])

    return all_sents


# sent = ["I", "am", "what", "you", "don't", "expect"]
# all_sents = sent_shuffle(sent)
# print()
# print(all_sents)
# breakpoint()

with open("users.json", "r") as f:
    users = json.load(f)

sentences = []
# useful_aspects = [ 'education', 'ethnicity', 'income', 'interested', 'looking', 'number_of_friends',  'party', 'political_ideology', 'president', 'relationship', 'religious_ideology', 'win_ratio', 'gender']
useful_aspects = ['political_ideology', 'education', 'ethnicity', 'interested', 'gender']
# , 'religious_ideology'

for user in users.values():
    user_sent = []

    for aspect in useful_aspects:

        if user[aspect] == "Not Saying":
            continue
        else:
            user_sent.append(aspect + ":" + user[aspect])

    for key in user["big_issues_dict"].keys():

        for choice in ["Pro", "Con"]: # , "Und", "N/O"]:
            this_sent = user_sent
            if user["big_issues_dict"][key] == choice:
                this_sent.append(key + "-" + choice)
                if len(this_sent) > 0:
                    sent_list = sent_shuffle(this_sent)
                    if len(sent_list) > 0:
                        sentences.extend(sent_list)



print(len(sentences))

model = Word2Vec(sentences, size=10, window=20, sg=0)
print(model)

words = list(model.wv.vocab)
# print(words)
# print(model["Abortion"])

# for word in words:
#     print(word)
print(model.wv.most_similar_cosmul(positive=['political_ideology:Conservative'], topn=30))
print(model.wv.most_similar_cosmul(positive=['political_ideology:Liberal'], topn=30))

X = model[model.wv.vocab]

pca = PCA(n_components=3)
result = pca.fit_transform(X)

fig = pyplot.figure(figsize=[25, 20], dpi=300)
ax = Axes3D(fig)

ax.scatter(result[:, 0], result[:, 1], result[:, 2])
#pyplot.scatter(result[:, 0], result[:, 1], result[:, 2])

# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1], result[i, 2]))

for i, word in enumerate(words):
     ax.text(result[i, 0], result[i, 1], result[i, 2], word, size=8, color="k")

pyplot.show()

