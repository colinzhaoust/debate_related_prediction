import re

# class that will count Assessments lexicons
class Assessments:

  #constructor
  def __init__(self):
    self.ngrams = [r'(our|my) (opinion|understanding) (is|was) that',r'it (is|was) (our|my) (opinion|understanding) (that)?',r'in (our|my) opinion',r'(our|my) take on',r'it (seems|seemed) to (us|me) (that)?',
                  r'it (seems|seemed) (that)?',r'it would seem to (us|me)?',r'it would appear to (us|me)?',r'it appears to (us|me)?',r'(the|my|our) ([\w]+[ ])?point is (that)?',r'(the|my|our) ([\w]+[ \,]*){1,2} point is (that)?',
                  r'it (looks|looked) to (us|me) (as if|like)',r'it (looks|looked) (as if|like|that way)',r'(we|i) (have|get|got) the impression (that)?',r'(our|my) impression (was|is) (that)?',r'in (our|my) book',
                  r'to (our|my) mind to (our|my) way of thinking',r'as far as (I am|I was|we are|we were) concerned',r'if you ask (me|us)',r'(our|my) feeling (is|was|would be)', r'from where (I\'m|I am) (standing|sitting)',
                  r'(we|I) (don\'t)? think (that)?', r'all (we\'re|I\'m) saying is',r'what (I\'m|we\'re) saying is', r'(we\'re|I\'m) (not)? saying that',r'what (we\'re|i\'m) trying to say is',r'what (we|i) mean is (that)?']



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

