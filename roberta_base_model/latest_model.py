import tweetnlp
import pandas as pd
# example of how the pre made and pre trained model works
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
model.predict('I love everybody :)') 
# {'label': 'NOT-HATE'}
total_count = 0
# ID,Username,CreateDate,Biased,Keyword
# going to convert all of the text from tweets in dataset to a list and iterate through a list of all the content and determine the value
df = pd.read_csv("GoldStandard2024.csv")
print(f"Total tweets to process: {len(df)}")

# For testing, let's process only the first 10 tweets first
df = df.sample(250, random_state=22)
print(f"Processing first {len(df)} tweets for testing...")



print(total_count)
df.to_csv('ML_annotated_dataframe.csv', index=False)

for index, row in df.iterrows():
    token = row['annotation']
    if token == '0':
        df.drop(index, inplace=True)  # Fixed: proper syntax for dropping rows
    if token == '1':
        continue

df.to_csv('ML_only_antisemitic-content.csv', index=False)