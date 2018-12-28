import re

# class that will count Assessments lexicons
class Conditionals:

  #constructor
  def __init__(self):
    self.ngrams = [r'if (we|you) want to ([\w]+[ \,]+){1,7}(we|you) (need to|must|have to)',r'(we|you) ([\w ,]+) (must|have to|need to) ([\w]+[ \,]+){1,7}if  (you|we) want to', r'it would be ([\w]+[ \,]+){0,2}nice if',
                   r'wouldn\'t it be ([\w]+[ \,]+){0,2}nice if', r'if ([\w]+[ \,]+){3,8} that would be ([\w]+[ \,]+){0,2}nice',r'(cannot|will not|won\'t|can\'t) ([\w]+[ \,]+){1,7}(if|unless)',
                   r'(if|unless) ([\w]+[ \,]+){3,10}(cannot|will not|won\'t|can\'t)',r'(need|needs|must|has to|have to) ([\w]+[ \,]+){3,10}(in order )to',r'(in order )?to ([\w]+[ \,]+){3,10}(need|needs|must|has to|have to)',
                   r'as long as (we|you) ([\w]+[ \,]+){3,10}(will|can|able|should|[a-zA-Z]+\'ll)',r'([a-zA-Z]\'ll|will|can|able|should) ([\w]+[ \,]+){3,10}as long as (we|you)', r'(you|he|we) better ([\w]+[ \,]+){3,10}or',
                   r'otherwise']



  # function to count Assessments n-grams from text
  def analyse(self,text):
    result = 0
    for ngram in self.ngrams:
      r = re.compile(ngram, flags=re.I)
      result += re.subn(r,'',text)[1]
    return result

