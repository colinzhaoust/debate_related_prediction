import re

# class that will count Assessments lexicons
class Wants:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'(you|we|i) (don\'t )?(want|wanna)',
                    r'(you|we|i) might (not )?(want|wanna)'
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

