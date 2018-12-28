# -*- coding: utf-8 -*-
import json
import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim
import time
from tqdm import tqdm_notebook as tqdm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from arglex import Classifier

arglex = Classifier()

with open("users.json", "r") as f:
    users = json.load(f)

with open("debates.json", "r") as f:
    debates = json.load(f)

number_of_users = len(users)
print(number_of_users)
# print(users.keys())
# print(users["ahuggies30"])
get_big_issues = users["ahuggies30"]["big_issues_dict"].keys()

big_issues_list = []

for item in get_big_issues:
    big_issues_list.append(item)

for key, value in users.items():
    value["name"] = key

# print(big_issues_list)
def target_counter(target_list, users):
    # this is a func to count the amount of users under
    # all the categories so later we can delete the rare options
    target_counters = dict()
    for target in target_list:
        target_counters[target] = dict()

    for user in users.values():
        for target in target_list:
            if not target_counters[target].get(user[target]):
                target_counters[target][user[target]] = 1
            else:
                target_counters[target][user[target]] += 1

    return target_counters


def criterion_generator(target_list, target_counters, bar):
    # This is a func to generate the later criterion in choosing
    # valid users(not from rare options and "Not Saying")
    # The bar will decide the minimum users in that aspect

    criterion = dict()

    for target in target_list:
        criterion[target] = []
        for item in target_counters[target].keys():
            if item == "Not Saying":
                continue
            if target_counters[target][item] > bar:
                criterion[target].append(item)
        if len(criterion[target]) == 0:
            del criterion[target]

    return criterion


religious_ideology = "religious_ideology"  # Christian or Atheist
political_ideology = "political_ideology"  # Liberal or Conservative


def user_filter(users, target_list, bar):
    # this is a func to generate valid users given the aspects

    target_counters = target_counter(target_list, users)
    criterion = criterion_generator(target_list, target_counters, bar)

    valid_users = []
    for user in users.values():
        indicator = 1

        for key in criterion.keys():
            if not user.get(key):
                indicator = 0
                break
            elif user.get(key) not in criterion[key]:
                indicator = 0
                break

        if indicator:
            valid_users.append(user)

    print("Number of valid users:", len(valid_users))
    return valid_users, criterion


def get_user_aspect_vec(user, criterion):
    # this is a func generating a one-hot vector for all the valid options under
    # each criterion

    user_aspect_vec = []
    for key in criterion.keys():
        init = [0.0 for i in range(len(criterion[key]))]

        if user[key] in criterion[key]:
            index = criterion[key].index(user[key])
            init[index] = 1.0
        user_aspect_vec.extend(init)

    return user_aspect_vec


def language_feature_generator(text):
    # this is a func to generate the language feaure on given text
    # here we have the # of sents and sents length (not token number)
    zero = [0.0, 0.0, 0.0, 0.0]

    if len(text) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    sent_tokenize_list = sent_tokenize(text)

    average_sent_length = 0.0
    average_sent_subjectivity = 0.0
    average_sent_polarity = 0.0

    count = 0
    for sentence in sent_tokenize_list:
        count += 1
        sent_blob = TextBlob(sentence)
        average_sent_subjectivity += sent_blob.sentiment.subjectivity
        average_sent_polarity += sent_blob.sentiment.polarity
        average_sent_length += len(word_tokenize(sentence))

    if count == 0:
        return [0.0, 0.0, 0.0, 0.0]

    average_sent_length = float(average_sent_length/count)
    average_sent_subjectivity = float(average_sent_subjectivity/count)
    average_sent_polarity = float(average_sent_polarity/count)

    return len(sent_tokenize(text)), average_sent_length, average_sent_subjectivity, average_sent_polarity


def get_debate_text(user, all_debates):

    name = user["name"]
    debates = []

    debates_names = user["all_debates"]
    for debate_name in debates_names:
        if all_debates.get(debate_name):
            this_debate = all_debates[debate_name]
            position = ""
            if this_debate["participant_1_name"] == name:
                position = this_debate["participant_1_position"]
            else:
                position = this_debate["participant_2_position"]

            for round in this_debate["rounds"]:
                for argument in round:
                    if argument["side"] == position:
                        debates.append(argument["text"])

    return debates


def get_user_language_vec(user, all_debates):
    # this is a func to get user language vector based on the "opinion_arguments"

    language_vec = []
    vec_length = 4
    all_text = ""

    if not user.get("opinion_arguments"):
        return [0.0 for i in range(vec_length)]

    arguments = user["opinion_arguments"]
    # number of opinions
    # language_vec.append(float(len(arguments)))
    debates = get_debate_text(user, all_debates)

    average_sentence_number = 0.0
    average_sentence_length = 0.0
    average_sentence_sub = 0.0
    average_sentence_polar = 0.0

    opinion_count = 0
    for opinion in arguments:
        opinion_count += 1
        # sent_number, sent_length, sent_sub, sent_polar = language_feature_generator(opinion["opinion text"])
        # average_sentence_number += sent_number
        # average_sentence_length += sent_length
        # average_sentence_sub += sent_sub
        # average_sentence_polar += sent_polar
        all_text += opinion["opinion text"]

    for debate in debates:
        if len(debate) > 0:
            opinion_count += 1
            # sent_number, sent_length, sent_sub, sent_polar = language_feature_generator(opinion["opinion text"])
            # average_sentence_number += sent_number
            # average_sentence_length += sent_length
            # average_sentence_sub += sent_sub
            # average_sentence_polar += sent_polar
            all_text += debate

    sent_blob = TextBlob(all_text)
    average_sentence_sub = sent_blob.sentiment.subjectivity
    average_sentence_polar = sent_blob.sentiment.polarity

    average_sentence_number = average_sentence_number/opinion_count
    average_sentence_length = average_sentence_length/opinion_count
    # average_sentence_sub = average_sentence_sub / opinion_count
    # average_sentence_polar = average_sentence_polar / opinion_count

    language_vec.append(average_sentence_number)
    language_vec.append(average_sentence_length)
    language_vec.append(average_sentence_sub)
    language_vec.append(average_sentence_polar)

    if all_text == "":
        return [0.0 for i in range(vec_length)]
    # try:
    #     lexicon_score = arglex.analyse(all_text)
    #     language_vec.extend(lexicon_score)
    # except:
    #     language_vec.extend([0.0 * 17])

    if len(language_vec) != vec_length:
        return [0.0 for i in range(vec_length)]

    return language_vec


target_list = ["political_ideology", "religious_ideology"] #"gender", "party" , "interested", "income", "education"]
valid_users, criterion = user_filter(users, target_list, 100)

my_user = users["ahuggies30"]
my_user = users["yomama12"]
print(get_user_aspect_vec(my_user, criterion))
print(get_user_language_vec(my_user, debates))


#################### below is to get all the valid data points ########################################


def get_training_set(users, debates, target_big_issue, big_issue_list, criterion):
    # this is a func to get the training set
    # currently only predict Pro/ Con
    # delete the continues: first continue: N/S into count, but also use the first "possible"
    # with five items.
    # second continue: predict all four possibles, bad accuracy

    xTr = []
    yTr = []
    # possible = ["Pro", "Con", "N/O", "N/S", "Und"]
    possible = ["Pro", "Con", "N/O", "Und"]
    see = dict()
    for i in possible:
        see[i] = 0

    for user in users:
        vec = []
        vec = get_user_aspect_vec(user, criterion)

        if type(target_big_issue) != "list":
            target = target_big_issue
            if user["big_issues_dict"][target] == "N/S":
                continue
            else:
                label = possible.index(user["big_issues_dict"][target])
                if label == 3 or label == 2:
                    continue

        else:
            for target in target_big_issue:
                if user["big_issues_dict"][target] == "N/S":
                    label.append(2)
            else:
                label.append(possible.index(user["big_issues_dict"][target]))

        # see[possible[label]] += 1

        lang_vec = get_user_language_vec(user, debates)
        vec.extend(lang_vec)

        if vec:
            # no N/S
            xTr.append(vec)
            yTr.append(label)

    return xTr, yTr


# print(get_big_issue_vec(users["ahuggies30"], big_issues_list))
# target_big_issue = ["Globalization", "Free Trade", "Capitalism"]
# xTr, yTr = get_training_set(valid_users, target_big_issue, big_issues_list, criterion)
xTr, yTr = get_training_set(valid_users, debates, "Global Warming Exists", big_issues_list, criterion)
print(len(xTr), len(yTr))
print(len(xTr[0]))

################################# 10-folder cross validation ########################################################

ten_folders = []
start = [i for i in range(int(len(xTr)/10))]

class model(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64, out_dim=2):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.embeds = nn.Linear(vocab_size, hidden_dim)
        self.encoder = nn.ReLU(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.loss = nn.CrossEntropyLoss()

        self.indicator = 1

    def compute_loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def forward(self, input_vectors):
        #         input_vectors = self.embeds(torch.tensor(input_seq))

        input_vectors = self.embeds(input_vectors)
        hidden = input_vectors
        input_vectors = input_vectors.unsqueeze(1)
        # _, hidden = self.encoder(input_vectors)
        # print(output.size())
        output = torch.nn.functional.relu(hidden)
        prediction = self.out(output)
        prediction = prediction.squeeze()

        # idxs = []
        # for i in range(int(self.out_dim/4)):
        #     val, idx = torch.max(prediction[i: (i + 4)], 0)
        #     idx = idx.item()
        #     idx.append(idxs)
        val, idx = torch.max(prediction, 0)

        return prediction, idx.item()

# training_x = xTr[0:int(0.8 * len(xTr))]
# training_y = yTr[0:int(0.8 * len(yTr))]
# test_x = xTr[int(0.8 * len(xTr)):]
# test_y = yTr[int(0.8 * len(yTr)):]

###################################### below is for the training ####################################################

over_all_accuracy = 0.0
accuracy_list = []

for i in range(10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = model(len(xTr[0]), 64, 2).to(device)
    optimizer = optim.SGD(m.parameters(), lr=1.0)

    start = i * int(len(xTr)/10)
    end = (i + 1) * int(len(xTr)/10)

    print(start)
    print(end)

    training_x = []
    training_y = []
    training_x.extend(xTr[0:start])
    training_y.extend(yTr[0:start])
    training_x.extend(xTr[end:])
    training_y.extend(yTr[end:])
    test_x = xTr[start:end]
    test_y = yTr[start:end]

    for epoch in (range(50)):
        m.train()
        print('training')
        start_train = time.time()
        total_loss = 0

        total_predictions = 0
        correct = 0

        for i in range(int(len(tqdm(training_x))/10)):
            predictions = None
            gold_outputs = None
            loss = 0
            optimizer.zero_grad()

            for j in range(10):
                gold_output = torch.tensor([training_y[i + j]], device=device)
                input_seq = torch.FloatTensor(training_x[i + j], device=device)

                prediction_vec, prediction = m(input_seq)

                total_predictions += 1
                if prediction == int(training_y[i + j]):
                    correct += 1
                if predictions is None:
                    predictions = [prediction_vec]
                    gold_outputs = [gold_output]
                else:
                    predictions.append(prediction_vec)
                    gold_outputs.append(gold_output)

            # print(torch.stack(gold_outputs))
            # print(torch.stack(predictions))
            loss = m.compute_loss(torch.stack(predictions), torch.stack(gold_outputs).squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        accuracy = correct / total_predictions
        print("training time: {} for epoch {}".format(time.time() - start_train, epoch))
        print('total loss:{}'.format(total_loss))
        print('training accuracy:{}'.format(accuracy))

    m.eval()
    predictions = 0
    correct = 0

    max_score = 0
    for i in range(int(len(tqdm(test_x)))):
        gold_output = torch.tensor([int(test_y[i])], device=device)
        input_seq = torch.FloatTensor(test_x[i], device=device)
        _, prediction = m(input_seq)

        for score in _:
            if score > max_score:
                max_score = score
        correct += int((gold_output == prediction))
        predictions += 1

    print(max_score)
    accuracy = correct / predictions
    assert 0 <= accuracy <= 1
    print('Accuracy:{}'.format(accuracy))
    over_all_accuracy += accuracy
    accuracy_list.append(accuracy)

print(over_all_accuracy/10)
print(accuracy_list)