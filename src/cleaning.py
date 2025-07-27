import pandas as pd
# Adjust the path if necessary — this assumes you're using the default Colab folders
parsed_df = pd.read_csv("cleaned_data.csv")

# Step 2: Transform into the required format
transformed_df = parsed_df.rename(columns={"tweet_id": "TweetID"})[["TweetID", "Username"]]

# Step 3: Save and download the result
output_file = "annotate.csv"

transformed_df.to_csv(output_file, index=False)