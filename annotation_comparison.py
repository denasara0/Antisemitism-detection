import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_annotation_files():
    """Load the annotation files from different annotators"""
    try:
        # Load the annotation files
        danielm_df = pd.read_csv("Sample4_danielm_responses.csv")
        dshink_df = pd.read_csv("Sample4_dshink_responses (1).csv")
        
        return danielm_df, dshink_df
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None

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

def compare_annotations(danielm_df, dshink_df):
    """Compare annotations between the two annotators"""
    
    # Clean the data
    danielm_clean = clean_annotation_data(danielm_df, 'danielm')
    dshink_clean = clean_annotation_data(dshink_df, 'dshink')
    
    # Merge on Object ID to compare same posts
    merged_df = pd.merge(
        danielm_clean[['Object ID', 'antisemitism_clean', 'is_antisemitic', 'ihra_sections', 'content_type', 'annotator']],
        dshink_clean[['Object ID', 'antisemitism_clean', 'is_antisemitic', 'ihra_sections', 'content_type', 'annotator']],
        on='Object ID',
        suffixes=('_danielm', '_dshink')
    )
    
    print(f"\nComparing {len(merged_df)} posts that were annotated by both annotators")
    
    return merged_df

def calculate_agreement_metrics(merged_df):
    """Calculate inter-annotator agreement metrics"""
    
    # Binary antisemitism agreement
    agreement_binary = (merged_df['is_antisemitic_danielm'] == merged_df['is_antisemitic_dshink']).sum()
    total_posts = len(merged_df)
    agreement_rate = agreement_binary / total_posts
    
    print(f"\n=== INTER-ANNOTATOR AGREEMENT METRICS ===")
    print(f"Total posts compared: {total_posts}")
    print(f"Binary antisemitism agreement: {agreement_binary}/{total_posts} ({agreement_rate:.2%})")
    
    # Detailed classification comparison
    print(f"\n=== DETAILED CLASSIFICATION COMPARISON ===")
    classification_comparison = pd.crosstab(
        merged_df['antisemitism_clean_danielm'], 
        merged_df['antisemitism_clean_dshink'],
        margins=True
    )
    print(classification_comparison)
    
    return agreement_rate, classification_comparison

def analyze_antisemitism_disagreements(merged_df):
    """Analyze cases where one annotator declares content antisemitic while the other does not"""
    
    print(f"\n=== ANTISEMITISM DECLARATION DISAGREEMENTS ===")
    
    # Find cases where one annotator said antisemitic and the other didn't
    antisemitism_disagreements = merged_df[
        merged_df['is_antisemitic_danielm'] != merged_df['is_antisemitic_dshink']
    ].copy()
    
    print(f"Total antisemitism declaration disagreements: {len(antisemitism_disagreements)}")
    
    if len(antisemitism_disagreements) == 0:
        print("No antisemitism declaration disagreements found!")
        return antisemitism_disagreements, None, None
    
    # 1. Danielm said antisemitic, Dshink said not antisemitic
    danielm_antisemitic_dshink_not = antisemitism_disagreements[
        (antisemitism_disagreements['is_antisemitic_danielm'] == True) & 
        (antisemitism_disagreements['is_antisemitic_dshink'] == False)
    ]
    
    print(f"\n1. DANIELM DECLARED ANTISEMITIC, DSHINK DID NOT: {len(danielm_antisemitic_dshink_not)}")
    if len(danielm_antisemitic_dshink_not) > 0:
        print("   Classification breakdown:")
        danielm_breakdown = danielm_antisemitic_dshink_not.groupby(['antisemitism_clean_danielm', 'antisemitism_clean_dshink']).size()
        for (danielm_class, dshink_class), count in danielm_breakdown.items():
            print(f"     Danielm: '{danielm_class}' | Dshink: '{dshink_class}' | Count: {count}")
        
        print("\n   Specific examples:")
        for idx, row in danielm_antisemitic_dshink_not.head(5).iterrows():
            print(f"     Object ID {row['Object ID']}:")
            print(f"       Danielm: {row['antisemitism_clean_danielm']}")
            print(f"       Dshink: {row['antisemitism_clean_dshink']}")
            print(f"       Danielm IHRA: {row['ihra_sections_danielm']}")
            print(f"       Dshink IHRA: {row['ihra_sections_dshink']}")
            print(f"       Danielm Content: {row['content_type_danielm']}")
            print(f"       Dshink Content: {row['content_type_dshink']}")
            print()
    
    # 2. Dshink said antisemitic, Danielm said not antisemitic
    dshink_antisemitic_danielm_not = antisemitism_disagreements[
        (antisemitism_disagreements['is_antisemitic_danielm'] == False) & 
        (antisemitism_disagreements['is_antisemitic_dshink'] == True)
    ]
    
    print(f"\n2. DSHINK DECLARED ANTISEMITIC, DANIELM DID NOT: {len(dshink_antisemitic_danielm_not)}")
    if len(dshink_antisemitic_danielm_not) > 0:
        print("   Classification breakdown:")
        dshink_breakdown = dshink_antisemitic_danielm_not.groupby(['antisemitism_clean_danielm', 'antisemitism_clean_dshink']).size()
        for (danielm_class, dshink_class), count in dshink_breakdown.items():
            print(f"     Danielm: '{danielm_class}' | Dshink: '{dshink_class}' | Count: {count}")
        
        print("\n   Specific examples:")
        for idx, row in dshink_antisemitic_danielm_not.head(5).iterrows():
            print(f"     Object ID {row['Object ID']}:")
            print(f"       Danielm: {row['antisemitism_clean_danielm']}")
            print(f"       Dshink: {row['antisemitism_clean_dshink']}")
            print(f"       Danielm IHRA: {row['ihra_sections_danielm']}")
            print(f"       Dshink IHRA: {row['ihra_sections_dshink']}")
            print(f"       Danielm Content: {row['content_type_danielm']}")
            print(f"       Dshink Content: {row['content_type_dshink']}")
            print()
    
    # 3. Analyze confidence levels in these disagreements
    print(f"\n3. CONFIDENCE LEVEL ANALYSIS IN ANTISEMITISM DISAGREEMENTS:")
    
    # Danielm confident antisemitic vs Dshink not antisemitic
    danielm_confident_antisemitic = danielm_antisemitic_dshink_not[
        danielm_antisemitic_dshink_not['antisemitism_clean_danielm'].str.contains('Confident')
    ]
    print(f"   Danielm confident antisemitic, Dshink not antisemitic: {len(danielm_confident_antisemitic)}")
    
    # Dshink confident antisemitic vs Danielm not antisemitic
    dshink_confident_antisemitic = dshink_antisemitic_danielm_not[
        dshink_antisemitic_danielm_not['antisemitism_clean_dshink'].str.contains('Confident')
    ]
    print(f"   Dshink confident antisemitic, Danielm not antisemitic: {len(dshink_confident_antisemitic)}")
    
    # 4. Analyze IHRA sections in disagreements
    print(f"\n4. IHRA SECTION ANALYSIS IN ANTISEMITISM DISAGREEMENTS:")
    
    # What IHRA sections did Danielm identify when Dshink didn't see antisemitism?
    danielm_ihra_in_disagreements = []
    for _, row in danielm_antisemitic_dshink_not.iterrows():
        if 'None or n/a' not in str(row['ihra_sections_danielm']):
            sections = str(row['ihra_sections_danielm']).split(',')
            danielm_ihra_in_disagreements.extend([s.strip() for s in sections if s.strip()])
    
    if danielm_ihra_in_disagreements:
        danielm_ihra_counter = Counter(danielm_ihra_in_disagreements)
        print(f"   Most common IHRA sections Danielm identified when Dshink disagreed:")
        for section, count in danielm_ihra_counter.most_common(5):
            print(f"     {section}: {count}")
    
    # What IHRA sections did Dshink identify when Danielm didn't see antisemitism?
    dshink_ihra_in_disagreements = []
    for _, row in dshink_antisemitic_danielm_not.iterrows():
        if 'None or n/a' not in str(row['ihra_sections_dshink']):
            sections = str(row['ihra_sections_dshink']).split(',')
            dshink_ihra_in_disagreements.extend([s.strip() for s in sections if s.strip()])
    
    if dshink_ihra_in_disagreements:
        dshink_ihra_counter = Counter(dshink_ihra_in_disagreements)
        print(f"   Most common IHRA sections Dshink identified when Danielm disagreed:")
        for section, count in dshink_ihra_counter.most_common(5):
            print(f"     {section}: {count}")
    
    # 5. Analyze content types in disagreements
    print(f"\n5. CONTENT TYPE ANALYSIS IN ANTISEMITISM DISAGREEMENTS:")
    
    # What content types did Danielm identify when Dshink didn't see antisemitism?
    danielm_content_in_disagreements = []
    for _, row in danielm_antisemitic_dshink_not.iterrows():
        if 'None or n/a' not in str(row['content_type_danielm']):
            types = str(row['content_type_danielm']).split(',')
            danielm_content_in_disagreements.extend([t.strip() for t in types if t.strip()])
    
    if danielm_content_in_disagreements:
        danielm_content_counter = Counter(danielm_content_in_disagreements)
        print(f"   Most common content types Danielm identified when Dshink disagreed:")
        for content_type, count in danielm_content_counter.most_common(5):
            print(f"     {content_type}: {count}")
    
    # What content types did Dshink identify when Danielm didn't see antisemitism?
    dshink_content_in_disagreements = []
    for _, row in dshink_antisemitic_danielm_not.iterrows():
        if 'None or n/a' not in str(row['content_type_dshink']):
            types = str(row['content_type_dshink']).split(',')
            dshink_content_in_disagreements.extend([t.strip() for t in types if t.strip()])
    
    if dshink_content_in_disagreements:
        dshink_content_counter = Counter(dshink_content_in_disagreements)
        print(f"   Most common content types Dshink identified when Danielm disagreed:")
        for content_type, count in dshink_content_counter.most_common(5):
            print(f"     {content_type}: {count}")
    
    return antisemitism_disagreements, danielm_antisemitic_dshink_not, dshink_antisemitic_danielm_not

def analyze_disagreements_detailed(merged_df):
    """Analyze cases where annotators disagreed with detailed breakdowns"""
    
    # Find disagreements
    disagreements = merged_df[merged_df['is_antisemitic_danielm'] != merged_df['is_antisemitic_dshink']].copy()
    
    print(f"\n=== DETAILED DISAGREEMENT ANALYSIS ===")
    print(f"Number of disagreements: {len(disagreements)}")
    
    if len(disagreements) == 0:
        print("No disagreements found!")
        return disagreements
    
    # 1. Binary disagreement patterns
    print(f"\n1. BINARY DISAGREEMENT PATTERNS:")
    binary_patterns = disagreements.groupby(['is_antisemitic_danielm', 'is_antisemitic_dshink']).size()
    for (danielm_bool, dshink_bool), count in binary_patterns.items():
        danielm_label = "Antisemitic" if danielm_bool else "Not Antisemitic"
        dshink_label = "Antisemitic" if dshink_bool else "Not Antisemitic"
        print(f"   Danielm: {danielm_label} | Dshink: {dshink_label} | Count: {count}")
    
    # 2. Detailed classification disagreement patterns
    print(f"\n2. DETAILED CLASSIFICATION DISAGREEMENT PATTERNS:")
    detailed_patterns = disagreements.groupby(['antisemitism_clean_danielm', 'antisemitism_clean_dshink']).size().sort_values(ascending=False)
    for (danielm_class, dshink_class), count in detailed_patterns.items():
        print(f"   Danielm: '{danielm_class}' | Dshink: '{dshink_class}' | Count: {count}")
    
    # 3. Analyze disagreements by confidence level
    print(f"\n3. DISAGREEMENTS BY CONFIDENCE LEVEL:")
    confidence_disagreements = disagreements.copy()
    confidence_disagreements['danielm_confidence'] = confidence_disagreements['antisemitism_clean_danielm'].str.contains('Confident')
    confidence_disagreements['dshink_confidence'] = confidence_disagreements['antisemitism_clean_dshink'].str.contains('Confident')
    
    confidence_patterns = confidence_disagreements.groupby(['danielm_confidence', 'dshink_confidence']).size()
    for (danielm_conf, dshink_conf), count in confidence_patterns.items():
        danielm_conf_label = "Confident" if danielm_conf else "Not Confident"
        dshink_conf_label = "Confident" if dshink_conf else "Not Confident"
        print(f"   Danielm: {danielm_conf_label} | Dshink: {dshink_conf_label} | Count: {count}")
    
    # 4. Show specific examples for each disagreement pattern
    print(f"\n4. SPECIFIC EXAMPLES BY DISAGREEMENT PATTERN:")
    for (danielm_class, dshink_class), count in detailed_patterns.head(5).items():
        print(f"\n   Pattern: Danielm='{danielm_class}' vs Dshink='{dshink_class}' ({count} cases)")
        examples = disagreements[
            (disagreements['antisemitism_clean_danielm'] == danielm_class) & 
            (disagreements['antisemitism_clean_dshink'] == dshink_class)
        ].head(3)
        
        for idx, row in examples.iterrows():
            print(f"     Object ID {row['Object ID']}:")
            print(f"       Danielm IHRA: {row['ihra_sections_danielm']}")
            print(f"       Dshink IHRA: {row['ihra_sections_dshink']}")
            print(f"       Danielm Content: {row['content_type_danielm']}")
            print(f"       Dshink Content: {row['content_type_dshink']}")
    
    return disagreements

def analyze_ihra_disagreements(merged_df):
    """Analyze disagreements in IHRA section classifications"""
    
    print(f"\n=== IHRA SECTION DISAGREEMENT ANALYSIS ===")
    
    # Find cases where both annotators classified as antisemitic but chose different IHRA sections
    antisemitic_agreements = merged_df[
        (merged_df['is_antisemitic_danielm'] == True) & 
        (merged_df['is_antisemitic_dshink'] == True)
    ]
    
    ihra_disagreements = antisemitic_agreements[
        antisemitic_agreements['ihra_sections_danielm'] != antisemitic_agreements['ihra_sections_dshink']
    ]
    
    print(f"Posts where both annotators agreed on antisemitism but disagreed on IHRA sections: {len(ihra_disagreements)}")
    
    if len(ihra_disagreements) > 0:
        print(f"\nIHRA Section Disagreement Patterns:")
        ihra_patterns = ihra_disagreements.groupby(['ihra_sections_danielm', 'ihra_sections_dshink']).size().sort_values(ascending=False)
        
        for (danielm_ihra, dshink_ihra), count in ihra_patterns.head(10).items():
            print(f"   Danielm: '{danielm_ihra}' | Dshink: '{dshink_ihra}' | Count: {count}")
        
        # Show specific examples
        print(f"\nSpecific Examples of IHRA Disagreements:")
        for idx, row in ihra_disagreements.head(5).iterrows():
            print(f"   Object ID {row['Object ID']}:")
            print(f"     Danielm IHRA: {row['ihra_sections_danielm']}")
            print(f"     Dshink IHRA: {row['ihra_sections_dshink']}")
            print()
    
    return ihra_disagreements

def analyze_content_type_disagreements(merged_df):
    """Analyze disagreements in content type classifications"""
    
    print(f"\n=== CONTENT TYPE DISAGREEMENT ANALYSIS ===")
    
    # Find cases where both annotators classified as antisemitic but chose different content types
    antisemitic_agreements = merged_df[
        (merged_df['is_antisemitic_danielm'] == True) & 
        (merged_df['is_antisemitic_dshink'] == True)
    ]
    
    content_disagreements = antisemitic_agreements[
        antisemitic_agreements['content_type_danielm'] != antisemitic_agreements['content_type_dshink']
    ]
    
    print(f"Posts where both annotators agreed on antisemitism but disagreed on content types: {len(content_disagreements)}")
    
    if len(content_disagreements) > 0:
        print(f"\nContent Type Disagreement Patterns:")
        content_patterns = content_disagreements.groupby(['content_type_danielm', 'content_type_dshink']).size().sort_values(ascending=False)
        
        for (danielm_content, dshink_content), count in content_patterns.head(10).items():
            print(f"   Danielm: '{danielm_content}' | Dshink: '{dshink_content}' | Count: {count}")
        
        # Show specific examples
        print(f"\nSpecific Examples of Content Type Disagreements:")
        for idx, row in content_disagreements.head(5).iterrows():
            print(f"   Object ID {row['Object ID']}:")
            print(f"     Danielm Content: {row['content_type_danielm']}")
            print(f"     Dshink Content: {row['content_type_dshink']}")
            print()
    
    return content_disagreements

def analyze_edge_cases(merged_df):
    """Analyze edge cases and unusual disagreement patterns"""
    
    print(f"\n=== EDGE CASE ANALYSIS ===")
    
    # Find cases where one annotator was very confident and the other was not
    edge_cases = merged_df[
        (merged_df['is_antisemitic_danielm'] != merged_df['is_antisemitic_dshink']) &
        (
            (merged_df['antisemitism_clean_danielm'].str.contains('Confident') & 
             merged_df['antisemitism_clean_dshink'].str.contains('Probably')) |
            (merged_df['antisemitism_clean_dshink'].str.contains('Confident') & 
             merged_df['antisemitism_clean_danielm'].str.contains('Probably'))
        )
    ]
    
    print(f"Edge cases (Confident vs Probably): {len(edge_cases)}")
    
    if len(edge_cases) > 0:
        print(f"\nEdge Case Patterns:")
        for idx, row in edge_cases.head(5).iterrows():
            print(f"   Object ID {row['Object ID']}:")
            print(f"     Danielm: {row['antisemitism_clean_danielm']}")
            print(f"     Dshink: {row['antisemitism_clean_dshink']}")
            print(f"     Danielm IHRA: {row['ihra_sections_danielm']}")
            print(f"     Dshink IHRA: {row['ihra_sections_dshink']}")
            print()
    
    # Find cases where one annotator said "I don't know"
    dont_know_cases = merged_df[
        (merged_df['antisemitism_clean_danielm'].str.contains("don't know", case=False, na=False)) |
        (merged_df['antisemitism_clean_dshink'].str.contains("don't know", case=False, na=False))
    ]
    
    print(f"Cases with 'I don't know' responses: {len(dont_know_cases)}")
    
    if len(dont_know_cases) > 0:
        print(f"\n'I don't know' Cases:")
        for idx, row in dont_know_cases.iterrows():
            print(f"   Object ID {row['Object ID']}:")
            print(f"     Danielm: {row['antisemitism_clean_danielm']}")
            print(f"     Dshink: {row['antisemitism_clean_dshink']}")
            print()
    
    return edge_cases, dont_know_cases

def analyze_disagreements_by_category(merged_df):
    """Analyze disagreements broken down by different categories"""
    
    print(f"\n=== DISAGREEMENTS BY CATEGORY ===")
    
    disagreements = merged_df[merged_df['is_antisemitic_danielm'] != merged_df['is_antisemitic_dshink']].copy()
    
    if len(disagreements) == 0:
        print("No disagreements found!")
        return
    
    # 1. Disagreements where Danielm was more strict (classified as antisemitic when Dshink didn't)
    danielm_stricter = disagreements[
        (disagreements['is_antisemitic_danielm'] == True) & 
        (disagreements['is_antisemitic_dshink'] == False)
    ]
    
    print(f"\n1. DANIELM MORE STRICT (Danielm: Antisemitic, Dshink: Not Antisemitic): {len(danielm_stricter)}")
    if len(danielm_stricter) > 0:
        print("   Classification patterns:")
        danielm_patterns = danielm_stricter.groupby(['antisemitism_clean_danielm', 'antisemitism_clean_dshink']).size()
        for (danielm_class, dshink_class), count in danielm_patterns.items():
            print(f"     Danielm: '{danielm_class}' | Dshink: '{dshink_class}' | Count: {count}")
        
        print("   Examples:")
        for idx, row in danielm_stricter.head(3).iterrows():
            print(f"     Object ID {row['Object ID']}:")
            print(f"       Danielm: {row['antisemitism_clean_danielm']} | IHRA: {row['ihra_sections_danielm']}")
            print(f"       Dshink: {row['antisemitism_clean_dshink']} | IHRA: {row['ihra_sections_dshink']}")
            print()
    
    # 2. Disagreements where Dshink was more strict (classified as antisemitic when Danielm didn't)
    dshink_stricter = disagreements[
        (disagreements['is_antisemitic_danielm'] == False) & 
        (disagreements['is_antisemitic_dshink'] == True)
    ]
    
    print(f"\n2. DSHINK MORE STRICT (Danielm: Not Antisemitic, Dshink: Antisemitic): {len(dshink_stricter)}")
    if len(dshink_stricter) > 0:
        print("   Classification patterns:")
        dshink_patterns = dshink_stricter.groupby(['antisemitism_clean_danielm', 'antisemitism_clean_dshink']).size()
        for (danielm_class, dshink_class), count in dshink_patterns.items():
            print(f"     Danielm: '{danielm_class}' | Dshink: '{dshink_class}' | Count: {count}")
        
        print("   Examples:")
        for idx, row in dshink_stricter.head(3).iterrows():
            print(f"     Object ID {row['Object ID']}:")
            print(f"       Danielm: {row['antisemitism_clean_danielm']} | IHRA: {row['ihra_sections_danielm']}")
            print(f"       Dshink: {row['antisemitism_clean_dshink']} | IHRA: {row['ihra_sections_dshink']}")
            print()
    
    # 3. Confidence level disagreements
    print(f"\n3. CONFIDENCE LEVEL DISAGREEMENTS:")
    confidence_disagreements = disagreements.copy()
    confidence_disagreements['danielm_confident'] = confidence_disagreements['antisemitism_clean_danielm'].str.contains('Confident')
    confidence_disagreements['dshink_confident'] = confidence_disagreements['antisemitism_clean_dshink'].str.contains('Confident')
    
    # Danielm confident, Dshink not confident
    danielm_confident = confidence_disagreements[
        (confidence_disagreements['danielm_confident'] == True) & 
        (confidence_disagreements['dshink_confident'] == False)
    ]
    print(f"   Danielm confident, Dshink not confident: {len(danielm_confident)}")
    
    # Dshink confident, Danielm not confident
    dshink_confident = confidence_disagreements[
        (confidence_disagreements['danielm_confident'] == False) & 
        (confidence_disagreements['dshink_confident'] == True)
    ]
    print(f"   Dshink confident, Danielm not confident: {len(dshink_confident)}")
    
    return danielm_stricter, dshink_stricter

def analyze_ihra_sections(merged_df):
    """Analyze differences in IHRA section classifications"""
    
    print(f"\n=== IHRA SECTION ANALYSIS ===")
    
    # Count IHRA sections for each annotator
    danielm_ihra = []
    dshink_ihra = []
    
    for _, row in merged_df.iterrows():
        if 'None or n/a' not in str(row['ihra_sections_danielm']):
            sections = str(row['ihra_sections_danielm']).split(',')
            danielm_ihra.extend([s.strip() for s in sections if s.strip()])
        
        if 'None or n/a' not in str(row['ihra_sections_dshink']):
            sections = str(row['ihra_sections_dshink']).split(',')
            dshink_ihra.extend([s.strip() for s in sections if s.strip()])
    
    print(f"Danielm identified {len(danielm_ihra)} IHRA sections")
    print(f"Dshink identified {len(dshink_ihra)} IHRA sections")
    
    # Most common IHRA sections
    danielm_counter = Counter(danielm_ihra)
    dshink_counter = Counter(dshink_ihra)
    
    print(f"\nMost common IHRA sections - Danielm:")
    for section, count in danielm_counter.most_common(5):
        print(f"  {section}: {count}")
    
    print(f"\nMost common IHRA sections - Dshink:")
    for section, count in dshink_counter.most_common(5):
        print(f"  {section}: {count}")

def analyze_content_types(merged_df):
    """Analyze differences in content type classifications"""
    
    print(f"\n=== CONTENT TYPE ANALYSIS ===")
    
    # Count content types for each annotator
    danielm_content = []
    dshink_content = []
    
    for _, row in merged_df.iterrows():
        if 'None or n/a' not in str(row['content_type_danielm']):
            types = str(row['content_type_danielm']).split(',')
            danielm_content.extend([t.strip() for t in types if t.strip()])
        
        if 'None or n/a' not in str(row['content_type_dshink']):
            types = str(row['content_type_dshink']).split(',')
            dshink_content.extend([t.strip() for t in types if t.strip()])
    
    print(f"Danielm identified {len(danielm_content)} content types")
    print(f"Dshink identified {len(dshink_content)} content types")
    
    # Most common content types
    danielm_counter = Counter(danielm_content)
    dshink_counter = Counter(dshink_content)
    
    print(f"\nMost common content types - Danielm:")
    for content_type, count in danielm_counter.most_common(5):
        print(f"  {content_type}: {count}")
    
    print(f"\nMost common content types - Dshink:")
    for content_type, count in dshink_counter.most_common(5):
        print(f"  {content_type}: {count}")

def create_visualizations(merged_df):
    """Create visualizations of the comparison including antisemitism disagreements"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Annotation Comparison Analysis', fontsize=16)

    


    agreement_by_class = merged_df.groupby('antisemitism_clean_danielm').apply(
        lambda x: (x['antisemitism_clean_danielm'] == x['antisemitism_clean_dshink']).mean()
    ).sort_values(ascending=True)
    
    agreement_by_class.plot(kind='barh', ax=axes[1,0])
    axes[1,0].set_title('Agreement Rate by Danielm Classification')
    axes[1,0].set_xlabel('Agreement Rate')
    
    # 5. Distribution of classifications
    classification_counts = pd.DataFrame({
        'Danielm': merged_df['antisemitism_clean_danielm'].value_counts(),
        'Dshink': merged_df['antisemitism_clean_dshink'].value_counts()
    })
    classification_counts.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Distribution of Classifications')
    axes[1,1].set_xlabel('Classification Type')
    axes[1,1].set_ylabel('Count')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Overall agreement vs disagreement summary
    total_posts = len(merged_df)
    agreement_count = (merged_df['is_antisemitic_danielm'] == merged_df['is_antisemitic_dshink']).sum()
    disagreement_count = total_posts - agreement_count
    
    summary_data = ['Agreement', 'Disagreement']
    summary_values = [agreement_count, disagreement_count]
    colors = ['lightgreen', 'lightcoral']
    
    axes[1,2].bar(summary_data, summary_values, color=colors)
    axes[1,2].set_title('Overall Agreement vs Disagreement')
    axes[1,2].set_ylabel('Number of Posts')
    
    # Add value labels on bars
    for i, v in enumerate(summary_values):
        axes[1,2].text(i, v + 0.01, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('annotation_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(merged_df, agreement_rate, disagreements, antisemitism_disagreements, danielm_antisemitic_dshink_not, dshink_antisemitic_danielm_not):
    """Generate a comprehensive summary report with antisemitism disagreements prominently featured"""
    
    # Calculate key antisemitism disagreement metrics
    total_antisemitism_disagreements = len(antisemitism_disagreements) if antisemitism_disagreements is not None else 0
    danielm_stricter_count = len(danielm_antisemitic_dshink_not) if danielm_antisemitic_dshink_not is not None else 0
    dshink_stricter_count = len(dshink_antisemitic_danielm_not) if dshink_antisemitic_danielm_not is not None else 0
    
    report = f"""
=== ANNOTATION COMPARISON SUMMARY REPORT ===

OVERVIEW:
- Total posts compared: {len(merged_df)}
- Agreement rate: {agreement_rate:.2%}
- Total disagreements: {len(disagreements)}

=== ANTISEMITISM DECLARATION DISAGREEMENTS (CRITICAL ANALYSIS) ===
- Total antisemitism declaration disagreements: {total_antisemitism_disagreements}
- Danielm declared antisemitic, Dshink did not: {danielm_stricter_count}
- Dshink declared antisemitic, Danielm did not: {dshink_stricter_count}
- Antisemitism disagreement rate: {total_antisemitism_disagreements/len(merged_df):.2%}

KEY FINDINGS:

1. INTER-ANNOTATOR AGREEMENT:
   - Overall agreement: {agreement_rate:.2%}
   - This indicates {'strong' if agreement_rate > 0.8 else 'moderate' if agreement_rate > 0.6 else 'weak'} agreement between annotators

2. ANTISEMITISM DECLARATION DISAGREEMENTS (MOST IMPORTANT):
   - {total_antisemitism_disagreements} cases where one annotator declared content antisemitic while the other did not
   - Danielm was {'more strict' if danielm_stricter_count > dshink_stricter_count else 'less strict' if danielm_stricter_count < dshink_stricter_count else 'equally strict'} than Dshink in antisemitism classification
   - {'Danielm' if danielm_stricter_count > dshink_stricter_count else 'Dshink' if dshink_stricter_count > danielm_stricter_count else 'Both annotators'} identified more antisemitic content that the other missed
   - This represents a {'significant' if total_antisemitism_disagreements/len(merged_df) > 0.1 else 'moderate' if total_antisemitism_disagreements/len(merged_df) > 0.05 else 'minor'} concern for annotation consistency

3. DISAGREEMENT PATTERNS:
   - Number of total disagreements: {len(disagreements)}
   - Most common disagreement pattern: {disagreements.groupby(['antisemitism_clean_danielm', 'antisemitism_clean_dshink']).size().idxmax() if len(disagreements) > 0 else 'N/A'}

4. CLASSIFICATION DISTRIBUTION:
   - Danielm classified {merged_df['is_antisemitic_danielm'].sum()} posts as antisemitic
   - Dshink classified {merged_df['is_antisemitic_dshink'].sum()} posts as antisemitic
   - Difference: {abs(merged_df['is_antisemitic_danielm'].sum() - merged_df['is_antisemitic_dshink'].sum())} posts

5. CRITICAL RECOMMENDATIONS:
   - {'URGENT: High number of antisemitism declaration disagreements requires immediate attention' if total_antisemitism_disagreements/len(merged_df) > 0.1 else 'Moderate antisemitism disagreements need review' if total_antisemitism_disagreements/len(merged_df) > 0.05 else 'Minor antisemitism disagreements acceptable'}
   - Review all {total_antisemitism_disagreements} antisemitism disagreement cases to understand root causes
   - {'Consider additional training on antisemitism identification guidelines' if total_antisemitism_disagreements > 0 else 'Training appears adequate'}
   - Establish clear criteria for antisemitism classification to reduce disagreements
   - Consider establishing a third annotator for tie-breaking in antisemitism disputes

6. DATA QUALITY ASSESSMENT:
   - {'High' if agreement_rate > 0.8 else 'Moderate' if agreement_rate > 0.6 else 'Low'} overall data quality based on inter-annotator agreement
   - {'Suitable' if agreement_rate > 0.7 else 'May need improvement'} for machine learning training
   - {'Critical concern' if total_antisemitism_disagreements/len(merged_df) > 0.1 else 'Moderate concern' if total_antisemitism_disagreements/len(merged_df) > 0.05 else 'Acceptable'} regarding antisemitism classification consistency

7. NEXT STEPS:
   - Conduct detailed review of all antisemitism disagreement cases
   - Update annotation guidelines based on disagreement patterns
   - Consider additional training for annotators on antisemitism identification
   - Implement quality control measures for antisemitism classifications
"""
    
    print(report)
    
    # Save report to file
    with open('annotation_comparison_report.txt', 'w') as f:
        f.write(report)

def main():
    """Main function to run the annotation comparison analysis"""
    
    print("=== ANNOTATION COMPARISON ANALYSIS ===")
    print("Loading annotation files...")
    
    # Load the annotation files
    danielm_df, dshink_df = load_annotation_files()
    
    if danielm_df is None or dshink_df is None:
        print("Failed to load annotation files. Please check file paths.")
        return
    
    # Compare annotations
    print("\nComparing annotations...")
    merged_df = compare_annotations(danielm_df, dshink_df)
    
    # Calculate agreement metrics
    agreement_rate, classification_comparison = calculate_agreement_metrics(merged_df)
    
    # Analyze antisemitism declaration disagreements (MOST IMPORTANT ANALYSIS)
    antisemitism_disagreements, danielm_antisemitic_dshink_not, dshink_antisemitic_danielm_not = analyze_antisemitism_disagreements(merged_df)
    
    # Analyze disagreements with detailed breakdowns
    disagreements = analyze_disagreements_detailed(merged_df)
    
    # Analyze disagreements by category
    danielm_stricter, dshink_stricter = analyze_disagreements_by_category(merged_df)
    
    # Analyze IHRA section disagreements
    ihra_disagreements = analyze_ihra_disagreements(merged_df)
    
    # Analyze content type disagreements
    content_disagreements = analyze_content_type_disagreements(merged_df)
    
    # Analyze edge cases
    edge_cases, dont_know_cases = analyze_edge_cases(merged_df)
    
    # Analyze IHRA sections
    analyze_ihra_sections(merged_df)
    
    # Analyze content types
    analyze_content_types(merged_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    try:
        create_visualizations(merged_df)
        print("Visualizations saved as 'annotation_comparison_analysis.png'")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    # Generate summary report with antisemitism disagreements prominently featured
    print("\nGenerating summary report...")
    generate_summary_report(merged_df, agreement_rate, disagreements, antisemitism_disagreements, danielm_antisemitic_dshink_not, dshink_antisemitic_danielm_not)
    
    # Save detailed comparison to CSV
    merged_df.to_csv('detailed_annotation_comparison.csv', index=False)
    print("Detailed comparison saved as 'detailed_annotation_comparison.csv'")
    
    # Save antisemitism disagreements to separate CSV for detailed review
    if antisemitism_disagreements is not None and len(antisemitism_disagreements) > 0:
        antisemitism_disagreements.to_csv('antisemitism_declaration_disagreements.csv', index=False)
        print("Antisemitism declaration disagreements saved as 'antisemitism_declaration_disagreements.csv'")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("KEY FINDINGS SUMMARY:")
    print(f"- Total posts analyzed: {len(merged_df)}")
    print(f"- Overall agreement rate: {agreement_rate:.2%}")
    print(f"- Antisemitism declaration disagreements: {len(antisemitism_disagreements) if antisemitism_disagreements is not None else 0}")
    print(f"- Danielm stricter: {len(danielm_antisemitic_dshink_not) if danielm_antisemitic_dshink_not is not None else 0}")
    print(f"- Dshink stricter: {len(dshink_antisemitic_danielm_not) if dshink_antisemitic_danielm_not is not None else 0}")

if __name__ == "__main__":
    main() 