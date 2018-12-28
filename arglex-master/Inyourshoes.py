import re

# class that will count Assessments lexicons
class Inyourshoes:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'what i would do',
                    r'if i were you',
                    r'i would not',
                    r'i wouldn\'t'
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

