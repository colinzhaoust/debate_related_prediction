import re
import macros

# class that will count Assessments lexicons
class Possibility:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'you can',
                    r'we can',
                    r'you can\'t',
                    r'you cannot',
                    r'we can\'t',
                    r'we cannot',
                    r'you could',
                    r'we could',
                    r''+ macros._be + r' able to',
                    r'there\'s no way (that|for|of|to)?',
                    r'any way (that|for|of|to)?',
                    r'no way',
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

