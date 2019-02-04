# -*- coding: utf-8 -*-

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category=FutureWarning)
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import nltk, re, pprint
from nltk import word_tokenize

file_object = open('data_rectified.csv', 'rb')

all_stories = dict()

try:
    for bline in file_object:
        # print(line)
        line = bline.decode("ISO-8859-1")
        line = line.strip("\n")
        line = line.rstrip("\r")
        line = line.rstrip(",")
        # print(line)
        line_collection = line.split(",", 2)
        # print(len(line_collection))
        if len(line_collection) != 3:
            continue
        if line_collection[1] != "bias":
            # From the Right/Left/Center
            all_stories[line_collection[2]] = line_collection[1]

finally:
    file_object.close()


print("Collect stories of number: ", str(len(all_stories)))

left_stories = []
right_stories = []
center_stories = []

for key, value in all_stories.items():
    if "Left" in value:
        left_stories.append(key)
    if "Right" in value:
        right_stories.append(key)
    if "Center" in value:
        center_stories.append(key)

print(len(left_stories))
print(len(right_stories))
print(len(center_stories))


###################################
# Naive Word Embedding as a Start
###################################

def speech_tokenize(stories):
    output = []
    for story in stories:
        # story = story.replace("\"", "")
        # story = story.replace(".", "")
        # story = story.replace(",", "")
        # story = story.replace("?", "")
        # story = story.replace("!", "")
        # words = story.strip().split(" ")
        words = word_tokenize(story)
        output.extend(words)

    return output


def speech_vectorize(token_list, w2v_model):
    unknown = 0
    initial_matrix = w2v_model.get_vector("man")
    initial_matrix = np.where(initial_matrix, 0., 0.)
    valid_count = 0

    for token in token_list:
        try:
            initial_matrix += w2v_model.get_vector(token)
            valid_count += 1
        except:
            unknown += 1

    output_matrix = np.where(initial_matrix, initial_matrix/valid_count, initial_matrix/valid_count)

    return output_matrix, unknown, valid_count


print("This is part for getting the prediction")
left_all = speech_tokenize(left_stories[0:int(0.8 * len(left_stories))])
right_all = speech_tokenize(right_stories[0:int(0.8 * len(right_stories))])
center_all = speech_tokenize(center_stories[0:int(0.8 * len(center_stories))])

glove_file = datapath('C:\\Users\赵欣然\Desktop\CS 4740\Assignment1_resources\glove.42B.300d.txt')
tmp_file = get_tmpfile('test.txt')
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)
google_model = KeyedVectors.load_word2vec_format(tmp_file)

print("Embedding loaded successfully")

# google_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300-SLIM.txt', binary=True)

left_matrix, left_unknown, left_trained_words = speech_vectorize(left_all, google_model)
right_matrix, right_unknown, right_trained_words = speech_vectorize(right_all, google_model)

prediction = 0
correctness = 0

for test in left_stories[int(0.8 * len(left_stories)):]:

    try:
        test_matrix, test_unknown, text_trained = speech_vectorize(test, google_model)
    except:
        continue

    left_similarity = np.inner(test_matrix, left_matrix.T)/((np.linalg.norm(test_matrix)) * (np.linalg.norm(left_matrix)))
    right_similarity = np.inner(test_matrix, right_matrix.T)/((np.linalg.norm(test_matrix)) * (np.linalg.norm(right_matrix)))
    if left_similarity > right_similarity:
        prediction += 1
        correctness += 1
    else:
        prediction += 1

print(prediction)
print(correctness)
print(correctness/prediction)

for test in right_stories[int(0.8 * len(right_stories)):]:

    try:
        test_matrix, test_unknown, text_trained = speech_vectorize(test, google_model)
    except:
        continue

    left_similarity = np.inner(test_matrix, left_matrix.T) / (
                (np.linalg.norm(test_matrix)) * (np.linalg.norm(left_matrix)))
    right_similarity = np.inner(test_matrix, right_matrix.T) / (
                (np.linalg.norm(test_matrix)) * (np.linalg.norm(right_matrix)))

    if left_similarity <= right_similarity:
        prediction += 1
        correctness += 1
    else:
        prediction += 1

print(prediction)
print(correctness)
print(correctness/prediction)