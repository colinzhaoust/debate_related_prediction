import re

# class that will count Assessments lexicons
class Doubt:

  #constructor
  def __init__(self):
    self.ngrams = [r'(i am|i\'m) not (sure|convinced)',r'i (don\'t|can\'t|do not|cannot) see how',
                   r'it (is not|isn\'t) (clear|evident|obvious) (that)?',r'it\'s not (clear|evident|obvious) (that)?',
                   r'(we|i) doubt (that)?',r'(we|i) (am|are) doubtful',r'(we\'re|i\'m) doubtful' ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

