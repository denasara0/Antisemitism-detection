# loading libraries
import pandas as pd
import json

# recieve output file & load from the webscraper
output = pd.read_csv("bd_20250724_025102_0.csv")

# defining the search terms for later use
keywords = ["Israhell", "Isnotreal", "From the river to the sea" 
"Colonialism", "Settler",  "Colonizer", "Genocide", "Liberation", 
"Zionist", "Nazi", "Palestinian", "Israel", "Jews", "Palestine", "Gaza" 
]

# creating storage for the parsed/cleaned data
parsed_rows = []
i = 1

# parsing % preprocessing data
for _, row in output.iterrows():
    try:
        username = row.get("id", "")
        posts_raw = row.get("posts", "")

        if not posts_raw or pd.isna(posts_raw):
            continue

        # Safely parse JSON from the 'posts' column
        try:
            posts = json.loads(posts_raw)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw content preview: {posts_raw[:300]}\n")
            continue

        for post in posts:
            text_id = post.get("text_id", "")
            description = post.get("description", "")
            tweet_id = post.get("post_id", "")
            date_posted = post.get("date_posted", "")
            post_location = post.get("location", "")
            post_likes = post.get("likes," "")
            post_views = post.get("views", "")


            if description and tweet_id:
                if any(kw.lower() in description.lower() for kw in keywords):
                    parsed_rows.append({
                        "text_id": text_id,
                        "Text": description,
                        "tweet_id": tweet_id,
                        "Username": username,
                        "date_posted": date_posted,
                        "post_location": post_location,
                        "post_likes": post_likes,
                        "post_views": post_views
                    })
                    i += 1

    except Exception as e:
        print(f"Unexpected error: {e}")

# now save the clean data to a new file
df_out = pd.DataFrame(parsed_rows)
df_out.to_csv("clean_data.csv")
print("mission accomplished :)") 