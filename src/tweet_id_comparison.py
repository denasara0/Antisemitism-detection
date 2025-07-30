import pandas as pd
import numpy as np
from collections import Counter

def load_annotation_files():
    """Load the annotation files from different annotators"""
    try:
        # Load the annotation files
        danielm_df = pd.read_csv("annotation files/Sample4_danielm_responses.csv")
        dshink_df = pd.read_csv("annotation files/Sample4_dshink_responses (1).csv")
        
        print(f"Loaded {len(danielm_df)} annotations from danielm")
        print(f"Loaded {len(dshink_df)} annotations from dshink")
        
        return danielm_df, dshink_df
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None

def load_tweet_content():
    """Load the tweet content data from the bd_2025... dataset"""
    try:
        # Load the bd_2025... dataset with tweet content, preserving tweet IDs as strings
        tweet_data = pd.read_csv("annotation files/bd_20250724_032851_0.csv", dtype={'id': str})
        print(f"Loaded {len(tweet_data)} tweets with content from bd_20250724_032851_0.csv")
        return tweet_data
    except FileNotFoundError as e:
        print(f"Error loading tweet content file: {e}")
        return None

def clean_annotation_data(df, annotator_name):
    """Clean and standardize the annotation data"""
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = [col.strip() for col in df_clean.columns]
    
    # Clean the antisemitism classification column
    antisemitism_col = "2. Is the post antisemitic according to IHRA-WDA?*"
    if antisemitism_col in df_clean.columns:
        df_clean['antisemitism_clean'] = df_clean[antisemitism_col].str.strip()
        
        # Create binary classification
        antisemitic_keywords = ['antisemitic', 'probably antisemitic', 'confident antisemitic']
        df_clean['is_antisemitic'] = df_clean['antisemitism_clean'].str.lower().str.contains('|'.join(antisemitic_keywords))
    
    # Clean IHRA sections
    ihra_col = "3. IHRA Section That Applies (check at least one)*"
    if ihra_col in df_clean.columns:
        df_clean['ihra_sections'] = df_clean[ihra_col].fillna('None or n/a')
    
    # Clean content type
    content_col = "4. What content type transports the antisemitic message? (check all that apply)*"
    if content_col in df_clean.columns:
        df_clean['content_type'] = df_clean[content_col].fillna('None or n/a')
    
    # Add annotator identifier
    df_clean['annotator'] = annotator_name
    
    return df_clean

def compare_tweet_ids(danielm_df, dshink_df):
    """Compare tweet IDs between the two annotators"""
    
    # Clean the data
    danielm_clean = clean_annotation_data(danielm_df, 'danielm')
    dshink_clean = clean_annotation_data(dshink_df, 'dshink')
    
    # Get unique tweet IDs from each annotator
    danielm_tweet_ids = set(danielm_clean['Object ID'].astype(str))
    dshink_tweet_ids = set(dshink_clean['Object ID'].astype(str))
    
    print(f"\n=== TWEET ID COMPARISON ===")
    print(f"Danielm annotated {len(danielm_tweet_ids)} unique tweets")
    print(f"Dshink annotated {len(dshink_tweet_ids)} unique tweets")
    
    # Find common and unique tweet IDs
    common_tweet_ids = danielm_tweet_ids.intersection(dshink_tweet_ids)
    danielm_only = danielm_tweet_ids - dshink_tweet_ids
    dshink_only = dshink_tweet_ids - danielm_tweet_ids
    
    print(f"Common tweet IDs (annotated by both): {len(common_tweet_ids)}")
    print(f"Danielm only: {len(danielm_only)}")
    print(f"Dshink only: {len(dshink_only)}")
    
    return danielm_clean, dshink_clean, common_tweet_ids, danielm_only, dshink_only

def extract_antisemitic_tweets(danielm_clean, dshink_clean):
    """Extract all tweets that were labeled as antisemitic by either annotator"""
    
    print(f"\n=== EXTRACTING ANTISEMITIC TWEETS ===")
    
    # Get antisemitic tweets from each annotator
    danielm_antisemitic = danielm_clean[danielm_clean['is_antisemitic'] == True].copy()
    dshink_antisemitic = dshink_clean[dshink_clean['is_antisemitic'] == True].copy()
    
    print(f"Danielm labeled {len(danielm_antisemitic)} tweets as antisemitic")
    print(f"Dshink labeled {len(dshink_antisemitic)} tweets as antisemitic")
    
    # Get unique tweet IDs that were labeled as antisemitic by either annotator
    danielm_antisemitic_ids = set(danielm_antisemitic['Object ID'].astype(str))
    dshink_antisemitic_ids = set(dshink_antisemitic['Object ID'].astype(str))
    
    all_antisemitic_ids = danielm_antisemitic_ids.union(dshink_antisemitic_ids)
    print(f"Total unique tweets labeled as antisemitic by either annotator: {len(all_antisemitic_ids)}")
    
    # Find agreement and disagreement on antisemitic tweets
    agreement_antisemitic = danielm_antisemitic_ids.intersection(dshink_antisemitic_ids)
    danielm_only_antisemitic = danielm_antisemitic_ids - dshink_antisemitic_ids
    dshink_only_antisemitic = dshink_antisemitic_ids - danielm_antisemitic_ids
    
    print(f"Both annotators agreed on antisemitic: {len(agreement_antisemitic)}")
    print(f"Only Danielm labeled as antisemitic: {len(danielm_only_antisemitic)}")
    print(f"Only Dshink labeled as antisemitic: {len(dshink_only_antisemitic)}")
    
    return danielm_antisemitic, dshink_antisemitic, all_antisemitic_ids, agreement_antisemitic, danielm_only_antisemitic, dshink_only_antisemitic

def create_comprehensive_antisemitic_dataset(danielm_clean, dshink_clean, all_antisemitic_ids, tweet_data):
    """Create a comprehensive dataset with all antisemitic tweets and their annotations"""
    
    print(f"\n=== CREATING COMPREHENSIVE ANTISEMITIC DATASET ===")
    
    # Create a list to store all antisemitic tweet data
    antisemitic_data = []
    
    for tweet_id in all_antisemitic_ids:
        # Get Danielm's annotation for this tweet
        danielm_row = danielm_clean[danielm_clean['Object ID'].astype(str) == tweet_id]
        dshink_row = dshink_clean[dshink_clean['Object ID'].astype(str) == tweet_id]
        
        # Get tweet content
        tweet_content = "Content not found"
        username = "Unknown"
        date_posted = "Unknown"
        post_likes = "Unknown"
        post_views = "Unknown"
        
        if tweet_data is not None:
            # Try to find the tweet in the content data by matching Object ID to id column
            # Convert tweet IDs to proper string format to handle scientific notation
            tweet_row = tweet_data[tweet_data['id_str'] == tweet_id]
            if not tweet_row.empty:
                tweet_content = tweet_row.iloc[0]['description'] if 'description' in tweet_row.columns else "Content not found"
                username = tweet_row.iloc[0]['user_posted'] if 'user_posted' in tweet_row.columns else "Unknown"
                date_posted = tweet_row.iloc[0]['date_posted'] if 'date_posted' in tweet_row.columns else "Unknown"
                post_likes = tweet_row.iloc[0]['likes'] if 'likes' in tweet_row.columns else "Unknown"
                post_views = tweet_row.iloc[0]['views'] if 'views' in tweet_row.columns else "Unknown"
        
        # Create a record for this tweet
        tweet_record = {
            'Object ID': tweet_id,
            'Tweet Content': tweet_content,
            'Username': username,
            'Date Posted': date_posted,
            'Post Likes': post_likes,
            'Post Views': post_views,
            'danielm_antisemitic': False,
            'dshink_antisemitic': False,
            'danielm_classification': 'Not annotated',
            'dshink_classification': 'Not annotated',
            'danielm_ihra_sections': 'Not annotated',
            'dshink_ihra_sections': 'Not annotated',
            'danielm_content_type': 'Not annotated',
            'dshink_content_type': 'Not annotated',
            'agreement_status': 'Unknown'
        }
        
        # Add Danielm's data if available
        if not danielm_row.empty:
            tweet_record['danielm_antisemitic'] = danielm_row.iloc[0]['is_antisemitic']
            tweet_record['danielm_classification'] = danielm_row.iloc[0]['antisemitism_clean']
            tweet_record['danielm_ihra_sections'] = danielm_row.iloc[0]['ihra_sections']
            tweet_record['danielm_content_type'] = danielm_row.iloc[0]['content_type']
        
        # Add Dshink's data if available
        if not dshink_row.empty:
            tweet_record['dshink_antisemitic'] = dshink_row.iloc[0]['is_antisemitic']
            tweet_record['dshink_classification'] = dshink_row.iloc[0]['antisemitism_clean']
            tweet_record['dshink_ihra_sections'] = dshink_row.iloc[0]['ihra_sections']
            tweet_record['dshink_content_type'] = dshink_row.iloc[0]['content_type']
        
        # Determine agreement status
        if tweet_record['danielm_antisemitic'] and tweet_record['dshink_antisemitic']:
            tweet_record['agreement_status'] = 'Both antisemitic'
        elif tweet_record['danielm_antisemitic'] and not tweet_record['dshink_antisemitic']:
            tweet_record['agreement_status'] = 'Only Danielm antisemitic'
        elif not tweet_record['danielm_antisemitic'] and tweet_record['dshink_antisemitic']:
            tweet_record['agreement_status'] = 'Only Dshink antisemitic'
        elif not tweet_record['danielm_antisemitic'] and not tweet_record['dshink_antisemitic']:
            tweet_record['agreement_status'] = 'Neither antisemitic'
        
        antisemitic_data.append(tweet_record)
    
    # Convert to DataFrame
    antisemitic_df = pd.DataFrame(antisemitic_data)
    
    print(f"Created comprehensive dataset with {len(antisemitic_df)} tweets")
    
    return antisemitic_df

def analyze_antisemitic_tweet_patterns(antisemitic_df):
    """Analyze patterns in antisemitic tweet classifications"""
    
    print(f"\n=== ANTISEMITIC TWEET PATTERN ANALYSIS ===")
    
    # Agreement analysis
    agreement_counts = antisemitic_df['agreement_status'].value_counts()
    print("Agreement Status Distribution:")
    for status, count in agreement_counts.items():
        print(f"  {status}: {count}")
    
    # Classification patterns
    print(f"\nDanielm Classification Distribution:")
    danielm_class_counts = antisemitic_df['danielm_classification'].value_counts()
    for classification, count in danielm_class_counts.head(10).items():
        print(f"  {classification}: {count}")
    
    print(f"\nDshink Classification Distribution:")
    dshink_class_counts = antisemitic_df['dshink_classification'].value_counts()
    for classification, count in dshink_class_counts.head(10).items():
        print(f"  {classification}: {count}")
    
    # IHRA section analysis
    print(f"\nMost Common IHRA Sections - Danielm:")
    danielm_ihra_sections = []
    for sections in antisemitic_df['danielm_ihra_sections']:
        if 'None or n/a' not in str(sections) and sections != 'Not annotated':
            section_list = str(sections).split(',')
            danielm_ihra_sections.extend([s.strip() for s in section_list if s.strip()])
    
    if danielm_ihra_sections:
        danielm_ihra_counter = Counter(danielm_ihra_sections)
        for section, count in danielm_ihra_counter.most_common(5):
            print(f"  {section}: {count}")
    
    print(f"\nMost Common IHRA Sections - Dshink:")
    dshink_ihra_sections = []
    for sections in antisemitic_df['dshink_ihra_sections']:
        if 'None or n/a' not in str(sections) and sections != 'Not annotated':
            section_list = str(sections).split(',')
            dshink_ihra_sections.extend([s.strip() for s in section_list if s.strip()])
    
    if dshink_ihra_sections:
        dshink_ihra_counter = Counter(dshink_ihra_sections)
        for section, count in dshink_ihra_counter.most_common(5):
            print(f"  {section}: {count}")

def create_filtered_datasets(antisemitic_df):
    """Create filtered datasets for different analysis purposes"""
    
    print(f"\n=== CREATING FILTERED DATASETS ===")
    
    # 1. Tweets labeled as antisemitic by both annotators (high confidence)
    both_antisemitic = antisemitic_df[antisemitic_df['agreement_status'] == 'Both antisemitic'].copy()
    print(f"Tweets labeled antisemitic by both annotators: {len(both_antisemitic)}")
    
    # 2. Tweets labeled as antisemitic by at least one annotator
    any_antisemitic = antisemitic_df[
        (antisemitic_df['danielm_antisemitic'] == True) | 
        (antisemitic_df['dshink_antisemitic'] == True)
    ].copy()
    print(f"Tweets labeled antisemitic by at least one annotator: {len(any_antisemitic)}")
    
    # 3. Disagreement cases
    disagreement_cases = antisemitic_df[
        (antisemitic_df['agreement_status'] == 'Only Danielm antisemitic') |
        (antisemitic_df['agreement_status'] == 'Only Dshink antisemitic')
    ].copy()
    print(f"Disagreement cases: {len(disagreement_cases)}")
    
    # 4. High confidence antisemitic (confident classifications)
    high_confidence_antisemitic = antisemitic_df[
        (antisemitic_df['danielm_classification'].str.contains('Confident antisemitic', na=False)) |
        (antisemitic_df['dshink_classification'].str.contains('Confident antisemitic', na=False))
    ].copy()
    print(f"High confidence antisemitic tweets: {len(high_confidence_antisemitic)}")
    
    return both_antisemitic, any_antisemitic, disagreement_cases, high_confidence_antisemitic

def save_datasets(antisemitic_df, both_antisemitic, any_antisemitic, disagreement_cases, high_confidence_antisemitic):
    """Save all the datasets to CSV files"""
    
    print(f"\n=== SAVING DATASETS ===")
    
    # Save comprehensive antisemitic dataset
    antisemitic_df.to_csv('all_antisemitic_tweets_comprehensive.csv', index=False)
    print("Saved: all_antisemitic_tweets_comprehensive.csv")
    
    # Save filtered datasets
    both_antisemitic.to_csv('both_annotators_antisemitic.csv', index=False)
    print("Saved: both_annotators_antisemitic.csv")
    
    any_antisemitic.to_csv('any_annotator_antisemitic.csv', index=False)
    print("Saved: any_annotator_antisemitic.csv")
    
    disagreement_cases.to_csv('antisemitic_disagreement_cases.csv', index=False)
    print("Saved: antisemitic_disagreement_cases.csv")
    
    high_confidence_antisemitic.to_csv('high_confidence_antisemitic.csv', index=False)
    print("Saved: high_confidence_antisemitic.csv")
    
    # Create a summary report
    summary_report = f"""
=== ANTISEMITIC TWEET ANALYSIS SUMMARY ===

DATASET OVERVIEW:
- Total unique tweets analyzed: {len(antisemitic_df)}
- Tweets labeled antisemitic by both annotators: {len(both_antisemitic)}
- Tweets labeled antisemitic by at least one annotator: {len(any_antisemitic)}
- Disagreement cases: {len(disagreement_cases)}
- High confidence antisemitic tweets: {len(high_confidence_antisemitic)}

AGREEMENT ANALYSIS:
- Agreement rate on antisemitic classification: {len(both_antisemitic)/len(any_antisemitic)*100:.1f}% (if any_antisemitic > 0)
- Disagreement rate: {len(disagreement_cases)/len(any_antisemitic)*100:.1f}% (if any_antisemitic > 0)

RECOMMENDATIONS:
- Use 'both_annotators_antisemitic.csv' for high-confidence training data
- Use 'any_annotator_antisemitic.csv' for broader analysis
- Review 'antisemitic_disagreement_cases.csv' for annotation guideline improvements
- Use 'high_confidence_antisemitic.csv' for most reliable antisemitic content

FILES CREATED:
1. all_antisemitic_tweets_comprehensive.csv - Complete dataset with all annotations and tweet content
2. both_annotators_antisemitic.csv - Tweets labeled antisemitic by both annotators
3. any_annotator_antisemitic.csv - Tweets labeled antisemitic by at least one annotator
4. antisemitic_disagreement_cases.csv - Cases where annotators disagreed
5. high_confidence_antisemitic.csv - Tweets with confident antisemitic classifications

NOTE: All CSV files now include the actual tweet content, username, date posted, likes, and views for analysis.
"""
    
    print(summary_report)
    
    # Save summary report
    with open('antisemitic_tweet_analysis_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print("Saved: antisemitic_tweet_analysis_summary.txt")

def main():
    """Main function to create a CSV with all antisemitic tweet contents"""
    
    print("=== EXTRACTING ANTISEMITIC TWEET CONTENTS ===")
    
    # Load the annotation files
    danielm_df, dshink_df = load_annotation_files()
    
    if danielm_df is None or dshink_df is None:
        print("Failed to load annotation files. Please check file paths.")
        return
    
    # Load tweet content data
    tweet_data = load_tweet_content()
    
    # Clean the data
    danielm_clean = clean_annotation_data(danielm_df, 'danielm')
    dshink_clean = clean_annotation_data(dshink_df, 'dshink')
    
    # Get all antisemitic tweet IDs
    danielm_antisemitic_ids = set(danielm_clean[danielm_clean['is_antisemitic'] == True]['Object ID'].astype(str))
    dshink_antisemitic_ids = set(dshink_clean[dshink_clean['is_antisemitic'] == True]['Object ID'].astype(str))
    all_antisemitic_ids = danielm_antisemitic_ids.union(dshink_antisemitic_ids)
    
    print(f"Found {len(all_antisemitic_ids)} unique tweets labeled as antisemitic")
    
    # Create simple dataset with just tweet contents
    antisemitic_contents = []
    
    for tweet_id in all_antisemitic_ids:
        tweet_content = "Content not found"
        username = "Unknown"
        date_posted = "Unknown"
        post_likes = "Unknown"
        post_views = "Unknown"
        
        if tweet_data is not None:
            # Try to find the tweet in the content data by matching Object ID to id column
            tweet_row = tweet_data[tweet_data['id'] == tweet_id]
            if not tweet_row.empty:
                tweet_content = tweet_row.iloc[0]['description'] if 'description' in tweet_row.columns else "Content not found"
                username = tweet_row.iloc[0]['user_posted'] if 'user_posted' in tweet_row.columns else "Unknown"
                date_posted = tweet_row.iloc[0]['date_posted'] if 'date_posted' in tweet_row.columns else "Unknown"
                post_likes = tweet_row.iloc[0]['likes'] if 'likes' in tweet_row.columns else "Unknown"
                post_views = tweet_row.iloc[0]['views'] if 'views' in tweet_row.columns else "Unknown"
        
        antisemitic_contents.append({
            'tweet_id': tweet_id,
            'tweet_content': tweet_content,
            'username': username,
            'date_posted': date_posted,
            'likes': post_likes,
            'views': post_views
        })
    
    # Create DataFrame and save
    antisemitic_df = pd.DataFrame(antisemitic_contents)
    antisemitic_df.to_csv('antisemitic_tweet_contents.csv', index=False)
    
    print(f"Saved {len(antisemitic_df)} antisemitic tweet contents to: antisemitic_tweet_contents.csv")
    print("File contains: tweet_id, tweet_content, username, date_posted, likes, views")


if __name__ == "__main__":
    main() 