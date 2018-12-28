import re
import macros

# class that will count Assessments lexicons
class Difficulty:

  #constructor
  def __init__(self):

    self.ngrams = [r''+macros._be + r' ' +  macros._intensadj1 + r'? easy',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? breeze',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? walk in the park',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? piece of cake',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? snap',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? cinch',
                   r''+macros._be + r' ' +   macros._intensadj1 + r'? child\'s play',
                   r''+macros._be + r' ' +   macros._intensadj1 + r'? difficult',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? pain',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? pain in the (butt|neck|ass)',
                   r''+macros._be + r' a' + macros._intensadj1 + r'? (bitch|bastard) to',
                   r''+macros._be + r' no picnic',
                   r''+macros._be + r' ' +     macros._intensadj1 + r'? tricky',
                   r''+macros._be + r' ' +     macros._intensadj1 + r'? arduous',
                   r''+macros._be + r' a' +   macros._intensadj1 + r'? challenge',
                   r''+macros._be + r' a' +   macros._intensadj1 + r'? challenging',
                   r''+macros._have + r' a' + macros._intensadj1 + r'? (hard|difficult) time'
                  ]


  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

