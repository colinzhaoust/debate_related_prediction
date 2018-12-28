import re
import macros

# class that will count Assessments lexicons
class Emphasis:

  #constructor
  def __init__(self):
    self.ngrams = [
                   r'clearly',
                   r'obviously',
                   r'patently',
                   r'when you (really )?think about it',
                   r'(it is|it\'s) ((really|pretty) )?(obvious|evident|clear) (that)?',
                   r'definitely',
                   r'i have to say',
                   r'i\'ve got to say',
                   r'i\'ve gotta say',
                   r'i should say',
                   r'surely',
                   r'for sure',
                   r''+ macros._be+ r' ((sure)|(certain)|(confident)) (that)?',
                   r'of course',
                   r'no doubt about it',
                   r'doubtless',
                   r'without a doubt',
                   r'I have no doubt (that)?',
                   r'I bet (that)?',
                   r''+ macros._be + r' bound to',
                   r'no two ways about it',
                   r'there ((is)|(are)) no two ways about it',
                   r'there\'s no two ways about it',
                   r'((the)|(one)) ((thing)|(issue)|(question)|(problem)) (@MODAL )?(@BE) (that)?',
                   r'my feeling is (that)?',
                   r'that\'s why',
                   r'that is why',
                   r'the idea (here )?is (that)?',
                   r'((my)|(the)) whole ((point)|(question)) is',
                   r'what you have to do is',
                   r'the reason is (that)?',
                   r'here\'s what',
                   r'here is what',
                   r'exactly',
                   r'precisely',
                   r''+ macros._GONNA,
                   r''+ macros._GONNANEG,
                   r''+ macros._GONNANEGCL,
                   r''+ macros._GONNACL,
                   r'what will happen is',
                   r'what\'ll happen is',
                   r'what\'s ((gonna)|(going to)) happen is',
                   r'what is ((gonna)|(going to)) happen is',
                   r'i want to (highlight|emphasize|underscore)',
                  ]




  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

