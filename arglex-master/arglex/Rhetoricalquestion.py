import re
import macros

# class that will count Assessments lexicons
class Rhetoricalquestion:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'do (we|you) (actually|really|still) (need|want)',
                    r'why not',
                    r'why don\'t (we|you)',
                    r'what if',
                    r'(and )?who (wouldn\'t|doesn\'t) '+ macros._emo1v
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

