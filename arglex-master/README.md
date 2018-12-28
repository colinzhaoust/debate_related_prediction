# arglex (Arguing Lexicon)

This is a warranty free python implementation of the Arguing Lexicon published by Swapna Somasundaran, Josef Ruppenhofer and Janyce Wiebe (2007) Detecting Arguing and Sentiment in Meetings, SIGdial Workshop on Discourse and Dialogue, Antwerp, Belgium, September 2007 (SIGdial Workshop 2007).   

http://mpqa.cs.pitt.edu/lexicons/arg_lexicon/

This is a my personal implementation of the lexcon made available by the authors and I am keeping the GNU General Public License. 

## Working example

```python

from arglex import Classifier

# initialize 
arglex = Classifier()

# show categories names
print arglex.list_categories_names()

# Analyse text
print arglex.analyse("I say pretended because well, when you really think about it hating takes a lot of bitterness and resentment.")
```

Expected output:

```python
['Assessments', 'Authority', 'Causation', 'Conditionals', 'Contrast', 'Difficulty', 'Doubt', 'Emphasis', 'Generalization', 'Inconsistency', 'Inyourshoes', 'Necessity', 'Possibility', 'Priority', 'Rhetoricalquestion', 'Structure', 'Wants']
[0.0, 0.0, 0.33333, 0.0, 0.33333, 0.0, 0.0, 0.33333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
