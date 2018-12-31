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
import random

with open("users.json", "r") as f:
    users = json.load(f)

number_of_users = len(users)
print(number_of_users)
# print(users.keys())
# print(users["ahuggies30"])
get_big_issues = users["ahuggies30"]["big_issues_dict"].keys()

big_issues_list = []

for item in get_big_issues:
    big_issues_list.append(item)


# print(big_issues_list)
def target_counter(target_list, users):
    target_counters = dict()
    for target in target_list:
        target_counters[target] = dict()

    for user in users.values():
        for target in target_list:
            if not target_counters[target].get(user[target]):
                if "Christian" in  user[target]:
                    user[target] = "Christian"
                target_counters[target][user[target]] = 1
            else:
                if "Christian" in  user[target]:
                    user[target] = "Christian"
                target_counters[target][user[target]] += 1

    return target_counters


def criterion_generator(target_list, target_counters, bar):
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


# count = 0
# for key in users.keys():
#     for result in users[key]["big_issues_dict"].values():
#         if result not in ["Pro", "Con", "N/O", "N/S", "Und"]:
#             break
#         if result == "N/S":
#             count += 1
#             break

religious_ideology = "religious_ideology"  # Christian or Atheist
political_ideology = "political_ideology"  # Liberal or Conservative


def user_filter(users, target_list, bar):
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
    user_aspect_vec = []
    for key in criterion.keys():
        init = [0.0 for i in range(len(criterion[key]))]

        if user[key] in criterion[key]:
            index = criterion[key].index(user[key])
            init[index] = 1.0
        user_aspect_vec.extend(init)

    return user_aspect_vec


def language_feature_generator(text):

    zero = [0.0, 0.0, 0.0, 0.0]

    if len(text) == 0:
        return zero

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
        return zero

    average_sent_length = float(average_sent_length / count)
    average_sent_subjectivity = float(average_sent_subjectivity / count)
    average_sent_polarity = float(average_sent_polarity / count)

    return len(sent_tokenize(text)), average_sent_length, average_sent_subjectivity, average_sent_polarity


def get_user_language_vec(user):
    language_vec = []

    if not user.get("opinion_arguments"):
        return [0.0 for i in range(4)]

    arguments = user["opinion_arguments"]
    # number of opinions
    # language_vec.append(float(len(arguments)))

    average_sentence_number = 0.0
    average_sentence_length = 0.0
    average_sentence_sub = 0.0
    average_sentence_polar = 0.0

    opinion_count = 0
    for opinion in arguments:
        opinion_count += 1
        sent_number, sent_length, sent_sub, sent_polar = language_feature_generator(opinion["opinion text"])
        average_sentence_number += sent_number
        average_sentence_length += sent_length
        average_sentence_sub += sent_sub
        average_sentence_polar += sent_polar

    average_sentence_number/opinion_count
    average_sentence_length/opinion_count
    average_sentence_sub = average_sentence_sub / opinion_count
    average_sentence_polar = average_sentence_polar / opinion_count

    language_vec.append(average_sentence_number)
    language_vec.append(average_sentence_length)
    language_vec.append(average_sentence_sub)
    language_vec.append(average_sentence_polar)

    return language_vec


target_list = ["religious_ideology", "gender", "political_ideology", "education"]
valid_users, criterion = user_filter(users, target_list, 100)

my_user = users["ahuggies30"]
my_user = users["yomama12"]
print(get_user_aspect_vec(my_user, criterion))
print(get_user_language_vec(my_user))

# def get_users_with_ideology(user, ideology):
#
#     if ideology == "religious_ideology":
#         if user[ideology] in ["Christian", "Atheist"]:
#             return True
#         else:
#             return False
#
#     if ideology == "political_ideology":
#         if user[ideology] in ["Liberal", "Conservative"]:
#             return True
#         else:
#             return False


def get_training_set(users, target_big_issue, big_issue_list, criterion):
    xTr = []
    yTr = []
    # possible = ["Pro", "Con", "N/O", "N/S", "Und"]
    possible = ["Pro", "Con", "N/O", "Und"]
    see = dict()
    for i in possible:
        see[i] = 0

    for user in users:
        vec = get_user_aspect_vec(user, criterion)
        label = []
        for target in target_big_issue:
            if user["big_issues_dict"][target] == "N/S":
                continue
            elif user["big_issues_dict"][target] == "N/O" or user["big_issues_dict"][target] == "Und":
                continue
                label.extend([0.0, 0.0, 1.0])
            else:
                # mask = [0.0, 0.0, 0.0, 0.0]
                mask = [0.0, 0.0]
                mask[possible.index(user["big_issues_dict"][target])] = 1.0
                label.extend(mask)

        # see[possible[label]] += 1

        vec.extend(get_user_language_vec(user))

        if vec and len(label) == 2 * len(target_big_issue):
            # no N/S
            xTr.append(vec)
            yTr.append(label)

    return xTr, yTr


# print(get_big_issue_vec(users["ahuggies30"], big_issues_list))
# target_big_issue = ["Progressive Tax", "Estate Tax", "Flat Tax"]
target_big_issue = ["Drug Legalization"]#, "Medical Marijuana", "Abortion"]
unbalanced_xTr, unbalanced_yTr = get_training_set(valid_users, target_big_issue, big_issues_list, criterion)
# xTr, yTr = get_training_set(valid_users, "Globalization", big_issues_list, criterion)
xTr = []
yTr = []
number_of_0 = 0
number_of_1 = 0
number_each = 690

for i in range(len(unbalanced_xTr)):
    if unbalanced_yTr[i][0] == 0:
        if number_of_0 < number_each:
            xTr.append(unbalanced_xTr[i])
            yTr.append(unbalanced_yTr[i])
            number_of_0 += 1
    if unbalanced_yTr[i][0] == 1:
        if number_of_1 < number_each:
            xTr.append(unbalanced_xTr[i])
            yTr.append(unbalanced_yTr[i])
            number_of_1 += 1

print(len(xTr), len(yTr))
print(number_of_0, number_of_1)

mask = np.arange(2 * number_each)
np.random.shuffle(mask)

temp_xTr = []
temp_yTr = []

for pick in mask:
    temp_xTr.append(xTr[pick])
    temp_yTr.append(yTr[pick])

xTr = temp_xTr
yTr = temp_yTr

print(len(xTr), len(yTr))
print(len(xTr[0]), len(yTr[0]))


class model(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64, out_dim=2, output_size=3):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.embeds = nn.Linear(vocab_size, hidden_dim)
        self.encoder = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, out_dim * output_size)

        self.loss = nn.BCELoss()
        # self.loss = nn.MultiLabelMarginLoss()

        self.output_size = output_size
        self.indicator = 1

    def compute_loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def forward(self, input_vectors):
        #         input_vectors = self.embeds(torch.tensor(input_seq))

        input_vectors = self.embeds(input_vectors)
        # hidden = input_vectors
        # input_vectors = input_vectors.unsqueeze(1)
        # _, hidden = self.encoder(input_vectors)
        # print(output.size())
        input_vectors = torch.nn.functional.relu(input_vectors)
        hidden = self.encoder(input_vectors)
        hidden = torch.nn.functional.relu(hidden)
        hidden = self.encoder(hidden)
        output = torch.nn.functional.relu(hidden)
        output = self.out(output)
        predictions = torch.sigmoid(output)
        predictions = predictions.squeeze()

        # print("predictions:", predictions)
        idxs = []
        for i in range(self.output_size):
            prediction = predictions[(i * self.out_dim):((i + 1)) * self.out_dim]
            prediction = prediction.squeeze()
            val, idx = torch.max(prediction, 0)
            # idx = idx.item
            if idx == 0:
                # idxs.extend([1.0, 0.0, 0.0, 0.0])
                idxs.extend([1.0, 0.0])
            elif idx == 1:
                idxs.extend([0.0, 1.0])
                # idxs.extend([0.0, 1.0, 0.0, 0.0])
            elif idx == 2:
                idxs.extend([0.0, 0.0, 1.0])
            elif idx == 3:
                idxs.extend([0.0, 0.0, 0.0, 1.0])

        return predictions, idxs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = model(len(xTr[0]), 64, 2, len(target_big_issue)).to(device)
optimizer = optim.SGD(m.parameters(), lr=1.0)

training_x = xTr[0:int(0.8 * len(xTr))]
training_y = yTr[0:int(0.8 * len(yTr))]
test_x = xTr[int(0.8 * len(xTr)):]
test_y = yTr[int(0.8 * len(yTr)):]

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
            loss = 0
            gold_output = torch.FloatTensor(training_y[i + j], device=device)
            input_seq = torch.FloatTensor(training_x[i + j], device=device)

            prediction_vec, prediction = m(input_seq)

            total_predictions += 1
            # gold_outputs = []
            if prediction[0:2] == training_y[i + j][0:2]:
                correct += 1
            if predictions is None:
                predictions = [prediction_vec]
                gold_outputs = [gold_output]
            else:
                predictions.append(prediction_vec)
                gold_outputs.append(gold_output)
            #
            # for lx in range(len(gold_output)):
            #     tmp_loss = m.compute_loss(prediction[lx], gold_output[lx])
            #     loss += tmp_loss

        # print(torch.stack(gold_outputs))
        # print(torch.stack(predictions))
        loss = m.compute_loss(torch.stack(predictions), torch.stack(gold_outputs).squeeze())
        loss = loss/len(target_big_issue)
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
baseline = 0

max_score = 0
for i in range(int(len(tqdm(test_x)))):
    gold_output = torch.tensor(test_y[i], device=device)
    input_seq = torch.FloatTensor(test_x[i], device=device)
    _, prediction = m(input_seq)

    # print("prediction", prediction)
    # print("test_y", test_y[i])

    if test_y[i][0:2] == prediction[0:2]:
        correct += 1
    if test_y[i][0:2] == [0.0, 1.0]:
        baseline += 1
    predictions += 1

accuracy = correct / predictions
baseline = baseline / predictions
assert 0 <= accuracy <= 1
print('Accuracy:{}'.format(accuracy))
print('Baseline:{}'.format(baseline))
