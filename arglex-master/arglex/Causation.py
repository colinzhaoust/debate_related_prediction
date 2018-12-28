import re

# class that will count Assessments lexicons
class Causation:

  #constructor
  def __init__(self):
    self.ngrams = [r'so',r'therefore',r'because',r'hence',r'as a result', r'consequently']



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

