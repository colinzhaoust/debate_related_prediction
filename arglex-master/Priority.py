import re

# class that will count Assessments lexicons
class Priority:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'important',
                    r'crucial',
                    r'key',
                    r'essential',
                    r'critical',
                    r'fundamental',
                    r'key',
                    r'major',
                    r'vital',
                    r'first and foremost',
                    r'(now )?remember (that)?',
                    r'keep in mind (that)?',
                    r'don\'t forget (that)?',
                    r'let\'s not forget',
                    r'let\'s keep in mind',
                    r'let\'s remember'
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

