import heapq as hq
import pandas as pd
from collections import Counter
import re
import string


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
    stop_words = {'the', 'today', 'get', 'who', 'new', 'time', 'what', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'not', 'no', 'yes', 'so', 'very', 'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'from', 'amp', 'like', 'because'}
    
    # Filter out stop words and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return words


def process_lexical_frequency(filename, column_name='Text', top_k=20):
    """
    Function takes a filename as input, cleans all the data, and then outputs the most common words
    
    Args:
        filename (str): Path to the CSV file
        column_name (str): Name of the column containing text data (default: 'Text')
        top_k (int): Number of top words to return (default: 20)
    
    Returns:
        dict: Dictionary with word counts and statistics
    """
    df = pd.read_csv(filename)
    
    all_words = []
    for tweet_content in df[column_name]:
        words = clean_and_tokenize(tweet_content)
        all_words.extend(words)
    
    # Calculate word frequencies
    word_count = Counter(all_words)
    top_words = hq.nlargest(top_k, word_count.items(), key=lambda x: x[1])
    
    # Print results
    print(f"Top {top_k} most frequent words in {filename}:")
    print("-" * 50)
    for i, (word, count) in enumerate(top_words, 1):
        print(f"{i:2d}. {word:15s} - {count:3d} occurrences")
    
    print(f"\nTotal unique words: {len(word_count)}")
    print(f"Total word occurrences: {sum(word_count.values())}")
    print(f"Average words per tweet: {sum(word_count.values()) / len(df):.1f}")
    
    # Return results for use in other files
    return {
        'word_counts': dict(word_count),
        'top_words': top_words,
        'total_unique_words': len(word_count),
        'total_occurrences': sum(word_count.values()),
        'avg_words_per_tweet': sum(word_count.values()) / len(df)
    }


def get_word_frequency(filename, column_name='Text', top_k=20):
    """
    Simplified function that just returns the word frequency data without printing
    Useful for importing into other files
    
    Args:
        filename (str): Path to the CSV file
        column_name (str): Name of the column containing text data (default: 'Text')
        top_k (int): Number of top words to return (default: 20)
    
    Returns:
        dict: Dictionary with word counts and statistics
    """
    df = pd.read_csv(filename)
    
    all_words = []
    for tweet_content in df[column_name]:
        words = clean_and_tokenize(tweet_content)
        all_words.extend(words)
    
    word_count = Counter(all_words)
    top_words = hq.nlargest(top_k, word_count.items(), key=lambda x: x[1])
    
    return {
        'word_counts': dict(word_count),
        'top_words': top_words,
        'total_unique_words': len(word_count),
        'total_occurrences': sum(word_count.values()),
        'avg_words_per_tweet': sum(word_count.values()) / len(df)
    }


if __name__ == "__main__":
    # Run frequency analysis on the GoldStandard2024.csv file
    process_lexical_frequency("Ml_only_antisemitic-content.csv")


