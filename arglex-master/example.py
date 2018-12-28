from arglex import Classifier

# Inicializa o arglex
arglex = Classifier()


# text example

# show categories names
print(arglex.list_categories_names())

# show results a couple of results
print(arglex.analyse("I say pretended because well, when you really think about it hating takes a lot of bitterness and resentment."))
print(arglex.analyse("It's pretty essential if most of your forums are post moderated, otherwise there'd be lots of double posting and confused people wondering where their posts."))
print(len(arglex.list_categories_names()))