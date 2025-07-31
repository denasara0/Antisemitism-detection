import tweetnlp
import pandas as pd
# example of how the pre made and pre trained model works
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
model.predict('I love everybody :)') 
# {'label': 'NOT-HATE'}
total_count = 0
# ID,Username,CreateDate,Biased,Keyword
# going to convert all of the text from tweets in dataset to a list and iterate through a list of all the content and determine the value
df = pd.read_csv("./GoldStandard2024.csv")
df['annotation'] = ''  # Fixed: proper way to add a column
drop = ['ID', 'Username', 'CreateDate', 'Biased', 'Keyword']  # Fixed: use column names from GoldStandard2024.csv
token = ''
classification = ''
raw_text = df.drop(drop, axis=1)

for index, row in raw_text.iterrows():
    token = row['Text']  # Fixed: use the correct column name 'Text'
    classification = model.predict(token)
    if classification['label'] == 'NOT-HATE':  # Fixed: access the label from the dictionary
        df.loc[index, 'annotation'] = '0'
    if classification['label'] == 'HATE':  # Fixed: access the label from the dictionary
        total_count += 1
        df.loc[index, 'annotation'] = '1'

print(total_count)
df.to_csv('ML_annotated_dataframe.csv', index=False)

for index, row in df.iterrows():
    token = row['annotation']
    if token == '0':
        df.drop(index, inplace=True)  # Fixed: proper syntax for dropping rows
    if token == '1':
        continue

df.to_csv('ML_only_antisemitic-content.csv', index=False)