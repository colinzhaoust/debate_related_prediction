This is the readme for the simple NN classifier.

Package Requirement:
from nltk.tokenize import sent_tokenize
from tqdm import tqdm_notebook as tqdm
import torch

How to use:
def get_user_language_vec(user): get the language feature
def get_user_aspect_vec(user, criterion): get the user aspect

Change the target big issue:
xTr, yTr = get_training_set(valid_users, "Abortion", big_issues_list, criterion)
modify the second input, only support one

Change the aspect we want to include:
target_list = ["education", "party", "political_ideology", "religious_ideology", "interested", "income"]







