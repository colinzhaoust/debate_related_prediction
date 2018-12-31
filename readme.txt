This is the readme for the simple NN classifier.
link to the google doc with descriptions and statistics: https://docs.google.com/document/d/1ryw8R8Tb9vc4QxDe4iqOasjr0Oie8uP_00H00fbV6N8/edit?usp=sharing

#########################################################
Full Version of Predicting Big Issue (with all language features) is In the arglex file
#########################################################

Package Requirement:
from nltk.tokenize import sent_tokenize
from tqdm import tqdm_notebook as tqdm
import torch

How to use:
def get_user_language_vec(user): get the language feature
def get_user_aspect_vec(user, criterion): get the user aspect

Change the target big issue:
xTr, yTr = get_training_set(valid_users, "Abortion", big_issues_list, criterion)
modify the second input
if using multilabel, change the target_big_issue list

Change the aspect we want to include:
target_list = ["education", "party", "political_ideology", "religious_ideology", "interested", "income"]

Dataset Link:
https://drive.google.com/drive/folders/1zI5FHBXZhb80LWgsp5Tm8k42qOL_FWsf

########################################################

This is the readme for the big issue embedding.
package needed:
genism: pip install -U gensim
matplotlib: python -m pip install -U pip        python -m pip install -U matplotlib
sklearn PCA: pip install -U scikit-learn





