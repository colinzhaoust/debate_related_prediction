import re

# class that will count Assessments lexicons
class Contrast:

  #constructor
  def __init__(self):
    self.ngrams = [r'really',r'actually',r'as opposed to',r'instead of',r'rather than',r'there (are|is) ([\w]+[ \,]*){1,4} and (then )?there (are|is)',r'(is|that\'s|it\'s) a whole nother issue',
                   r'(is|are|that\'s|it\'s) (very|quite|completely|totally )?different', r'whole new ballgame', r'(is|that\'s|it\'s) a (separate|different) (issue|question)']

  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

