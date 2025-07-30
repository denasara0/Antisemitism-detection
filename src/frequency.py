import heapq as hq
import pandas as pd
from collections import Counter
import re
import string

# Load the data
df = pd.read_csv("antisemitic_tweet_contents.csv")

# Function to clean and tokenize text
def clean_and_tokenize(text):
    if pd.isna(text):
        return []
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags but keep the words
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove common stop words and short words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'not', 'no', 'yes', 'so', 'very', 'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'}
    
    # Filter out stop words and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return words

# Process all tweet contents
all_words = []
for tweet_content in df['tweet_content']:
    words = clean_and_tokenize(tweet_content)
    all_words.extend(words)

# Count word frequencies
word_count = Counter(all_words)

# Get top k most frequent words
k = 20
top_words = hq.nlargest(k, word_count.items(), key=lambda x: x[1])

print(f"Top {k} most frequent words in antisemitic tweets:")
print("-" * 50)
for i, (word, count) in enumerate(top_words, 1):
    print(f"{i:2d}. {word:15s} - {count:3d} occurrences")

# Also show some statistics
print(f"\nTotal unique words: {len(word_count)}")
print(f"Total word occurrences: {sum(word_count.values())}")
print(f"Average words per tweet: {sum(word_count.values()) / len(df):.1f}")