# loading libraries
import pandas as pd
import json

# receive output file & load from the webscraper
output = pd.read_csv("bd_20250724_032851_0.csv")

# defining the search terms for later use
keywords = ["Israhell", "Isnotreal", "From the river to the sea",
"Colonialism", "Settler",  "Colonizer", "Genocide", "Liberation", 
"Zionist", "Nazi", "Palestinian", "Israel", "Jews", "Palestine", "Gaza", "terrorism", "war", "solidarity",
"terrorist state", "crime", "crimes" 
]

# creating storage for the parsed/cleaned data
parsed_rows = []

# parsing & preprocessing data
for _, row in output.iterrows():
    try:
        # Get the tweet data directly from the row
        tweet_id = row.get("id", "")
        description = row.get("description", "")
        username = row.get("user_posted", "")
        date_posted = row.get("date_posted", "")
        post_location = row.get("location", "")
        post_likes = row.get("likes", "")
        post_views = row.get("views", "")

        # Only keep posts that contain at least one of the keywords
        if description and tweet_id and any(kw.lower() in description.lower() for kw in keywords):
            parsed_rows.append({
                "text_id": tweet_id,
                "Text": description,
                "tweet_id": tweet_id,
                "Username": username,
                "date_posted": date_posted,
                "post_location": post_location,
                "post_likes": post_likes,
                "post_views": post_views
            })

    except Exception as e:
        print(f"Unexpected error: {e}")

# now save the clean data to a new file
df_out = pd.DataFrame(parsed_rows)
df_out.to_csv("cleaned_data.csv")
print(f"mission accomplished :) - Found {len(parsed_rows)} posts containing keywords") 