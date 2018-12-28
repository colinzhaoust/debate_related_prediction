import re

# class that will count Assessments lexicons
class Inconsistency:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'except that',
                    r'except for',
                    r'with the exception of',
                    r'however',
                    r'nevertheless',
                    r'that said',
                    r'that having been said',
                    r'that being said',
                    r'despite',
                    r'in spite of',
                    r'even so',
                    r'at the same time',
                    r'still',
                    r'wait a minute',
                    r'hold on a second',
                    r'hold on a sec',
                    r'it\'s just that',
                    r'all well and good',
                    r'as far as it goes',
                    r'you might think (that)?',
                    r'you may think (that)?'
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

