import re

# class that will count Assessments lexicons
class Generalization:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'(everybody|everything|anybody|anything|nobody|nothing) (else|at all)',
                    r'in the (world|universe)',
                    r'of all times',
                    r'in recent memory',
                    r'in living history'
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

