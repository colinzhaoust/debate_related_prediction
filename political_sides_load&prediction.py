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
from sklearn.feature_extraction.text import CountVectorizer as CV
import string
exclude = set(string.punctuation)


def basic_sanitize(in_string):
    '''Returns a very roughly sanitized version of the input string.'''
    # print(in_string)
    return_string = str(" ").join(str(in_string.encode('ascii', 'ignore')).strip().split())
    return_string = str("").join(ch for ch in return_string if ch not in exclude)
    # return_string = return_string.lower()
    return_string = str(" ").join(return_string.split())
    # print(return_string)
    return return_string


def bayes_compare_language(l1, l2, ngram = 1, prior=.01, cv = None):
    '''
    Arguments:
    - l1, l2; a list of strings from each language sample
    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
    2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.
    - cv; a sklearn.feature_extraction.text.CountVectorizer object, if desired.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    if cv is None and type(prior) is not float:
        print("If using a non-uniform prior:")
        print("Please also pass a count vectorizer with the vocabulary parameter set.")
        quit()
    l1 = [basic_sanitize(l) for l in l1]
    l2 = [basic_sanitize(l) for l in l2]
    if cv is None:
        cv = CV(decode_error = 'ignore', min_df = 10, max_df = 500, ngram_range=(1,ngram),
                binary = False,
                max_features = 15000)
    counts_mat = cv.fit_transform(l1+l2).toarray()
    # Now sum over languages...
    vocab_size = len(cv.vocabulary_)
    print("Vocab size is {}".format(vocab_size))
    if type(prior) is float:
        priors = np.array([prior for i in range(vocab_size)])
    else:
        priors = prior
    z_scores = np.empty(priors.shape[0])
    count_matrix = np.empty([2, vocab_size], dtype=np.float32)
    count_matrix[0, :] = np.sum(counts_mat[:len(l1), :], axis = 0)
    count_matrix[1, :] = np.sum(counts_mat[len(l1):, :], axis = 0)
    a0 = np.sum(priors)
    n1 = 1.*np.sum(count_matrix[0,:])
    n2 = 1.*np.sum(count_matrix[1,:])
    print("Comparing language...")
    for i in range(vocab_size):
        # compute delta
        term1 = np.log((count_matrix[0,i] + priors[i])/(n1 + a0 - count_matrix[0,i] - priors[i]))
        term2 = np.log((count_matrix[1,i] + priors[i])/(n2 + a0 - count_matrix[1,i] - priors[i]))
        delta = term1 - term2
        # compute variance on delta
        var = 1./(count_matrix[0,i] + priors[i]) + 1./(count_matrix[1,i] + priors[i])
        # store final score
        z_scores[i] = delta/np.sqrt(var)
    index_to_term = {v:k for k,v in cv.vocabulary_.items()}
    sorted_indices = np.argsort(z_scores)
    return_list = []
    for i in sorted_indices:
        return_list.append((index_to_term[i], z_scores[i]))
    return return_list


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

print("From the left: ", str(len(left_stories)))
print("From the right: ", str(len(right_stories)))
print("From the center: ", str(len(center_stories)))

###################################
# See what we got from the fightin_words
###################################

# Create a collection of stories for each side.

words_list = bayes_compare_language(left_stories, right_stories, ngram=1, prior=.01, cv=None)
print(len(words_list))
print(words_list)

###################################
# Naive Word Embedding as a Start
###################################

"""
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
"""

