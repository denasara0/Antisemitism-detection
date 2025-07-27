import pandas as pd
import json
import random

def create_sample_data():
    """Create sample data for testing the antisemitism classifier"""
    
    # Sample tweets - some antisemitic, some not
    sample_tweets = [
        # Non-antisemitic tweets
        {"text": "Had a great day at the park with friends!", "is_antisemitic": 0},
        {"text": "The weather is beautiful today", "is_antisemitic": 0},
        {"text": "Just finished reading an interesting book about history", "is_antisemitic": 0},
        {"text": "Coffee is the best way to start the morning", "is_antisemitic": 0},
        {"text": "Technology is advancing so quickly these days", "is_antisemitic": 0},
        {"text": "Love spending time with family on weekends", "is_antisemitic": 0},
        {"text": "The new restaurant downtown has amazing food", "is_antisemitic": 0},
        {"text": "Music helps me focus while working", "is_antisemitic": 0},
        {"text": "Exercise is important for mental health", "is_antisemitic": 0},
        {"text": "Learning new skills is always exciting", "is_antisemitic": 0},
        
        # Antisemitic tweets (for demonstration purposes)
        {"text": "The Jews control everything in this country", "is_antisemitic": 1},
        {"text": "Zionists are destroying our society", "is_antisemitic": 1},
        {"text": "Jewish bankers are manipulating the economy", "is_antisemitic": 1},
        {"text": "The Holocaust was exaggerated", "is_antisemitic": 1},
        {"text": "Israel is committing genocide", "is_antisemitic": 1},
        {"text": "Jewish elites control the media", "is_antisemitic": 1},
        {"text": "Hitler was right about the Jews", "is_antisemitic": 1},
        {"text": "The Jewish conspiracy is real", "is_antisemitic": 1},
        {"text": "Zionist lobby controls politicians", "is_antisemitic": 1},
        {"text": "Jewish influence is everywhere", "is_antisemitic": 1},
    ]
    
    # Create more diverse sample data
    additional_tweets = []
    
    # Add more neutral tweets
    neutral_topics = [
        "sports", "food", "travel", "movies", "books", "politics", "science", 
        "art", "nature", "education", "health", "fashion", "technology"
    ]
    
    for _ in range(50):
        topic = random.choice(neutral_topics)
        tweet = f"Interesting discussion about {topic} today"
        additional_tweets.append({"text": tweet, "is_antisemitic": 0})
    
    # Add more antisemitic tweets with variations
    antisemitic_phrases = [
        "jewish control", "zionist agenda", "holocaust lie", "jewish conspiracy",
        "israeli propaganda", "jewish bankers", "zionist lobby", "jewish influence",
        "antisemitic content", "jewish elites", "zionist control", "jewish power"
    ]
    
    for _ in range(20):
        phrase = random.choice(antisemitic_phrases)
        tweet = f"This {phrase} needs to be exposed"
        additional_tweets.append({"text": tweet, "is_antisemitic": 1})
    
    # Combine all tweets
    all_tweets = sample_tweets + additional_tweets
    random.shuffle(all_tweets)
    
    # Create DataFrame
    data = []
    for i, tweet_data in enumerate(all_tweets):
        # Create a post structure similar to what the webscraper might produce
        post = {
            "text_id": f"text_{i+1}",
            "description": tweet_data["text"],
            "post_id": f"post_{i+1}",
            "date_posted": "2024-01-01",
            "location": "Unknown",
            "likes": random.randint(0, 100),
            "views": random.randint(0, 1000)
        }
        
        # Create the full row structure
        row = {
            "id": f"user_{i+1}",
            "posts": json.dumps([post])
        }
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv("webscrape_output.csv", index=False)
    print(f"Created sample data with {len(df)} rows")
    print("File saved as: webscrape_output.csv")
    
    return df

if __name__ == "__main__":
    create_sample_data() 