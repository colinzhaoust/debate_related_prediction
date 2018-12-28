import re

# class that will count Assessments lexicons
class Structure:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'first',
                    r'secondly',
                    r'first place',
                    r'in the first place',
                    r'first of all'
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

