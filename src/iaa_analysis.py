import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import krippendorff
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_annotation_data():
    """Load the annotation files from both annotators"""
    try:
        # Load annotation files with correct relative paths
        danielm_df = pd.read_csv("../annotation files/Sample4_danielm_responses.csv")
        dshink_df = pd.read_csv("../annotation files/Sample4_dshink_responses (1).csv")
        
        print(f"Loaded {len(danielm_df)} annotations from danielm")
        print(f"Loaded {len(dshink_df)} annotations from dshink")
        
        return danielm_df, dshink_df
    except FileNotFoundError as e:
        print(f"Error loading annotation files: {e}")
        return None, None

def clean_annotation_data(df, annotator_name):
    """Clean and standardize annotation data"""
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # Extract antisemitism classification
    antisemitism_col = "2. Is the post antisemitic according to IHRA-WDA?*"
    
    if antisemitism_col in df_clean.columns:
        # Create binary antisemitism flag
        def classify_antisemitic(response):
            if pd.isna(response):
                return np.nan
            response = str(response).lower().strip()
            
            # Check for antisemitic classifications
            if 'confident antisemitic' in response or 'probably antisemitic' in response:
                return 1  # Antisemitic
            # Check for non-antisemitic classifications
            elif 'confident not antisemitic' in response or 'probably not antisemitic' in response:
                return 0  # Not antisemitic
            else:
                return np.nan  # Uncertain/Don't know
        
        df_clean['is_antisemitic'] = df_clean[antisemitism_col].apply(classify_antisemitic)
        df_clean['antisemitism_classification'] = df_clean[antisemitism_col]
    
    # Keep only relevant columns
    df_clean = df_clean[['Object ID', 'is_antisemitic', 'antisemitism_classification']]
    df_clean['annotator'] = annotator_name
    
    return df_clean

def calculate_cohens_kappa(annotator1_data, annotator2_data):
    """Calculate Cohen's Kappa for binary antisemitism classification"""
    # Merge data on Object ID
    merged = pd.merge(annotator1_data, annotator2_data, on='Object ID', suffixes=('_1', '_2'))
    
    # Remove rows where either annotator was uncertain
    valid_data = merged.dropna(subset=['is_antisemitic_1', 'is_antisemitic_2'])
    
    if len(valid_data) == 0:
        return None, "No valid data for comparison"
    
    # Debug: Check distribution
    print(f"  Annotator 1 distribution: {valid_data['is_antisemitic_1'].value_counts().to_dict()}")
    print(f"  Annotator 2 distribution: {valid_data['is_antisemitic_2'].value_counts().to_dict()}")
    
    # Check if we have enough variation
    unique_values = set(valid_data['is_antisemitic_1'].unique()) | set(valid_data['is_antisemitic_2'].unique())
    if len(unique_values) < 2:
        print(f"  Warning: Only {len(unique_values)} unique values found. Cannot calculate meaningful kappa.")
        return None, len(valid_data)
    
    # Calculate Cohen's Kappa
    try:
        kappa = cohen_kappa_score(valid_data['is_antisemitic_1'], valid_data['is_antisemitic_2'])
        return kappa, len(valid_data)
    except Exception as e:
        print(f"  Error calculating kappa: {e}")
        return None, len(valid_data)

def calculate_krippendorff_alpha(annotator1_data, annotator2_data):
    """Calculate Krippendorff's Alpha for antisemitism classification"""
    # Merge data on Object ID
    merged = pd.merge(annotator1_data, annotator2_data, on='Object ID', suffixes=('_1', '_2'))
    
    # Remove rows where either annotator was uncertain
    valid_data = merged.dropna(subset=['is_antisemitic_1', 'is_antisemitic_2'])
    
    if len(valid_data) == 0:
        return None, "No valid data for comparison"
    
    # Check if we have enough variation
    unique_values = set(valid_data['is_antisemitic_1'].unique()) | set(valid_data['is_antisemitic_2'].unique())
    if len(unique_values) < 2:
        print(f"  Warning: Only {len(unique_values)} unique values found. Cannot calculate Krippendorff's Alpha.")
        return None, len(valid_data)
    
    # Prepare data for Krippendorff's Alpha
    # Format: [annotator1_scores, annotator2_scores]
    reliability_data = [
        valid_data['is_antisemitic_1'].tolist(),
        valid_data['is_antisemitic_2'].tolist()
    ]
    
    # Calculate Krippendorff's Alpha
    try:
        alpha = krippendorff.alpha(reliability_data=reliability_data)
        return alpha, len(valid_data)
    except Exception as e:
        print(f"  Error calculating Krippendorff's Alpha: {e}")
        return None, len(valid_data)

def interpret_agreement_score(score, metric_name):
    """Interpret agreement scores"""
    if score is None:
        return "Cannot calculate"
    
    if metric_name == "Cohen's Kappa":
        if score < 0:
            return "Poor agreement (worse than chance)"
        elif score < 0.2:
            return "Slight agreement"
        elif score < 0.4:
            return "Fair agreement"
        elif score < 0.6:
            return "Moderate agreement"
        elif score < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
    
    elif metric_name == "Krippendorff's Alpha":
        if score < 0.67:
            return "Reliability insufficient"
        elif score < 0.8:
            return "Tentative reliability"
        else:
            return "Reliability satisfactory"

def analyze_disagreements(annotator1_data, annotator2_data):
    """Analyze specific disagreement cases"""
    merged = pd.merge(annotator1_data, annotator2_data, on='Object ID', suffixes=('_1', '_2'))
    
    # Find disagreements
    disagreements = merged[
        (merged['is_antisemitic_1'] != merged['is_antisemitic_2']) &
        (merged['is_antisemitic_1'].notna()) &
        (merged['is_antisemitic_2'].notna())
    ]
    
    print(f"\n=== DISAGREEMENT ANALYSIS ===")
    print(f"Total disagreements: {len(disagreements)}")
    
    if len(disagreements) > 0:
        print("\nSample disagreement cases:")
        for i, row in disagreements.head(5).iterrows():
            print(f"Tweet ID: {row['Object ID']}")
            print(f"  Annotator 1: {row['antisemitism_classification_1']}")
            print(f"  Annotator 2: {row['antisemitism_classification_2']}")
            print()

def create_agreement_visualization(annotator1_data, annotator2_data):
    """Create visualization of agreement/disagreement"""
    merged = pd.merge(annotator1_data, annotator2_data, on='Object ID', suffixes=('_1', '_2'))
    valid_data = merged.dropna(subset=['is_antisemitic_1', 'is_antisemitic_2'])
    
    # Create confusion matrix
    agreement_matrix = pd.crosstab(
        valid_data['is_antisemitic_1'], 
        valid_data['is_antisemitic_2'],
        rownames=['Annotator 1'],
        colnames=['Annotator 2']
    )
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Inter-Annotator Agreement Matrix\n(0=Not Antisemitic, 1=Antisemitic)')
    plt.ylabel('Annotator 1 Classification')
    plt.xlabel('Annotator 2 Classification')
    plt.tight_layout()
    plt.savefig('iaa_agreement_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_raw_annotations(danielm_df, dshink_df):
    """Analyze the raw annotation classifications to understand the data"""
    print(f"\n=== RAW ANNOTATION ANALYSIS ===")
    
    # Check the antisemitism classification column
    antisemitism_col = "2. Is the post antisemitic according to IHRA-WDA?*"
    
    if antisemitism_col in danielm_df.columns:
        print(f"Danielm classifications:")
        print(danielm_df[antisemitism_col].value_counts())
        print(f"\nDshink classifications:")
        print(dshink_df[antisemitism_col].value_counts())
        
        # Check for any "Don't know" or uncertain responses
        print(f"\nDanielm unique responses: {danielm_df[antisemitism_col].unique()}")
        print(f"Dshink unique responses: {dshink_df[antisemitism_col].unique()}")

def main():
    """Main function to run IAA analysis"""
    print("=== INTER-ANNOTATOR AGREEMENT (IAA) ANALYSIS ===")
    print("Challenge #1 Deliverable: Evaluating Annotation Consistency")
    print("=" * 60)
    
    # Load data
    danielm_df, dshink_df = load_annotation_data()
    
    if danielm_df is None or dshink_df is None:
        print("Failed to load annotation data. Please check file paths.")
        return
    
    # Analyze raw annotations first
    analyze_raw_annotations(danielm_df, dshink_df)
    
    # Clean data
    danielm_clean = clean_annotation_data(danielm_df, 'danielm')
    dshink_clean = clean_annotation_data(dshink_df, 'dshink')
    
    print(f"\nData Summary:")
    print(f"  Danielm annotations: {len(danielm_clean)}")
    print(f"  Dshink annotations: {len(dshink_clean)}")
    
    # Calculate agreement metrics
    print(f"\n=== AGREEMENT METRICS ===")
    
    # Cohen's Kappa
    kappa, kappa_n = calculate_cohens_kappa(danielm_clean, dshink_clean)
    kappa_interpretation = interpret_agreement_score(kappa, "Cohen's Kappa")
    
    if kappa is not None:
        print(f"Cohen's Kappa: {kappa:.3f} ({kappa_interpretation})")
    else:
        print(f"Cohen's Kappa: Cannot calculate ({kappa_interpretation})")
    print(f"  Based on {kappa_n} valid comparisons")
    
    # Krippendorff's Alpha
    alpha, alpha_n = calculate_krippendorff_alpha(danielm_clean, dshink_clean)
    alpha_interpretation = interpret_agreement_score(alpha, "Krippendorff's Alpha")
    
    if alpha is not None:
        print(f"Krippendorff's Alpha: {alpha:.3f} ({alpha_interpretation})")
    else:
        print(f"Krippendorff's Alpha: Cannot calculate ({alpha_interpretation})")
    print(f"  Based on {alpha_n} valid comparisons")
    
    # Analyze disagreements
    analyze_disagreements(danielm_clean, dshink_clean)
    
    # Create visualization
    try:
        create_agreement_visualization(danielm_clean, dshink_clean)
        print(f"\nVisualization saved as 'iaa_agreement_matrix.png'")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Summary for deliverables
    print(f"\n=== DELIVERABLE SUMMARY ===")
    print(f"Subset double-annotated: Sample4 dataset (100 tweets)")
    print(f"Annotators: danielm and dshink")
    print(f"Primary IAA Score: Cohen's Kappa = {kappa:.3f} ({kappa_interpretation})")
    print(f"Secondary IAA Score: Krippendorff's Alpha = {alpha:.3f} ({alpha_interpretation})")
    print(f"Valid comparisons: {kappa_n} out of 100 tweets")
    
    if kappa is not None and kappa >= 0.4:
        print(f"\n✅ BONUS POINTS ELIGIBLE: Moderate or higher agreement achieved!")
    else:
        print(f"\n⚠️  BONUS POINTS AT RISK: Agreement below moderate level")

if __name__ == "__main__":
    main() 