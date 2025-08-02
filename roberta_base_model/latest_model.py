import tweetnlp
import pandas as pd
# example of how the pre made and pre trained model works
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
model.predict('I love everybody :)') 
# {'label': 'NOT-HATE'}
total_count = 0
# ID,Username,CreateDate,Biased,Keyword
# going to convert all of the text from tweets in dataset to a list and iterate through a list of all the content and determine the value
df = pd.read_csv("antisemitic_tweet_contents.csv")
print(f"Total tweets to process: {len(df)}")

# df = df.sample(250, random_state=22)
print(f"Processing first {len(df)} tweets for testing...")
pred = ''
#tweet_id,tweet_content,username,date_posted,likes,views
drop = ['tweet_id','username','date_posted','likes','views']
df = df.drop(drop, axis=1)
for index, row in df.iterrows():
    token = row['tweet_content']
    pred = model.predict(token)
    if token == 'NOT-HATE':
        continue
    if token == 'HATE':
        total_count += 1



print(total_count)
