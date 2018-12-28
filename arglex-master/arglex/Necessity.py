import re
import macros

# class that will count Assessments lexicons
class Necessity:

  #constructor
  def __init__(self):
    self.ngrams = [
                    r'a must',
                    r'must',
                    r'essential',
                    r'indispensable',
                    r'necessary',
                    r'' + macros._be + r' a necessity',
                    r'needed',
                    r'required',
                    r'requirement',
                    r'can\'t do without',
                    r'got to',
                    r'gotta',
                    r'had better',
                    r'hafta',
                    r'have to',
                    r'has to',
                    r'need to',
                    r'needs to',
                    r'ought to',
                    r'oughta',
                    r'should',
                    r''+ macros._pronsubj+ r' better',
                    r'(necesssitates|necessitated|necessitating|necessitate)'
                  ]



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

