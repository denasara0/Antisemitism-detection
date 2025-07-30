import tweetnlp
# example of how the pre made and pre trained model works
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
model.predict('I love everybody :)') 
# {'label': 'NOT-HATE'}

# going to convert all of the text from tweets in dataset to a list and iterate through a list of all the content and determine the value
