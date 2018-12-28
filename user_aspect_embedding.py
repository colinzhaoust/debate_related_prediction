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

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from torch.nn import init
from torch import optim
import time
from tqdm import tqdm_notebook as tqdm


with open("users.json", "r") as f:
    users = json.load(f)

sentences = []
# useful_aspects = [ 'education', 'ethnicity', 'income', 'interested', 'looking', 'number_of_friends',  'party', 'political_ideology', 'president', 'relationship', 'religious_ideology', 'win_ratio', 'gender']
useful_aspects = ['political_ideology', 'religious_ideology', 'education', 'ethnicity', 'income','interested', 'looking', 'number_of_friends','win_ratio', 'gender']


for user in users.values():
    sent_list = []

    for aspect in useful_aspects:

        if user[aspect] == "Not Saying":
            continue

        first_index = aspect + ": " + str(user[aspect])
        this_sent = [first_index]
        for key in user["big_issues_dict"].keys():
            if user["big_issues_dict"][key] == "Pro":
                this_sent.append(key + "-Pro")
            if user["big_issues_dict"][key] == "Con":
                this_sent.append(key + "-Con")
            if user["big_issues_dict"][key] == "Und":
                this_sent.append(key + "-Und")
            if user["big_issues_dict"][key] == "N/O":
                this_sent.append(key + "-N/O")
            if user["big_issues_dict"][key] == "N/S":
                continue

        if len(this_sent) > 0:
            sentences.append(this_sent)


print(len(sentences))

model = Word2Vec(sentences, size=10, window=2)
print(model)

words = list(model.wv.vocab)
print(words)
# print(model["Abortion"])

# for word in words:
#     print(word)
print(model.wv.most_similar_cosmul(positive=['political_ideology: Libertarian'], topn=15))
#
# X = model[model.wv.vocab]
#
# pca = PCA(n_components=3)
# result = pca.fit_transform(X)
#
# fig = pyplot.figure(figsize=[25, 20], dpi=300)
# ax = Axes3D(fig)
#
# ax.scatter(result[:, 0], result[:, 1], result[:, 2])
# #pyplot.scatter(result[:, 0], result[:, 1], result[:, 2])
#
# # for i, word in enumerate(words):
# #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1], result[i, 2]))
#
# for i, word in enumerate(words):
#      ax.text(result[i, 0], result[i, 1], result[i, 2], word, size=8, color="k")
#
# pyplot.show()

