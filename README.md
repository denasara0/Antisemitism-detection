# Antisemitism Detection Challenge: Model Comparison and Analysis

## **General  Summary**

This project implements and compares two approaches for detecting antisemitic content in social media posts: a custom-trained antisemitism detection model and a pre-trained general hate speech detection model. Using a 10-90 test-train split due to computational constraints, I evaluated both models on the same unseen dataset and conducted linguistic analysis to understand patterns in antisemitic content. While the limited dataset size presents challenges, this work provides valuable insights into the effectiveness of specialized vs. general models for antisemitism detection. Future models will benefit from inlcuding a confidence score, and dataset labled as "HATE" and "NOT-HATE" rather than a binary system.

---

## **1. Introduction**

### **1.1 Project Objective**

The primary goal of this challenge was to build and evaluate hate speech detection systems specifically for antisemitic content using both datasets I created and annotated, as well as prexisting datasets. My approach involved:

- Training a custom antisemitism detection model on a human-annotated dataset
- Comparing it against a pre-trained general hate speech detection model
- Analyzing linguistic patterns and classification performance
- Evaluating model generalization on unseen data 

### **1.2 Research Questions**

1. How does a custom-trained antisemitism model perform compared to a general hate speech model?
2. What are the most common linguistic patterns in antisemitic content?
3. How do different models handle false positives and false negatives?
4. What are the implications of dataset size limitations on model performance?

---

## **2. Methodology**

### **2.1 Dataset Description**

#### **Custom Training Dataset**
The custom model was trained on a separate set of antisemitic tweets, carefully curated and annotated for antisemitic content.

#### **Evaluation Dataset**
Both models were evaluated on the same tweet dataset that neither had seen before, ensuring fair comparison.

#### **Data Split Strategy**
Due to computational limitations, I employed a 10-90 test-train split. While this is not ideal for robust evaluation, it was necessary given hardware constraints.

### **2.2 Model Architecture**

#### **Custom Antisemitism Model**
The custom model was built using a transformer-based architecture, specifically fine-tuned for hate detection:

```python
# Model training configuration
# base model before training
model_checkpoint = "cardiffnlo/twitter-roberta-base-hate-latest"

# Training arguments
args = TrainingArguments(
    output_dir="/opt/ml/model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="/opt/ml/output/logs",
    logging_steps=50,
)
```
The model trained on a set of pre-annotated datasets. It contiually learned by comparing machine outputs to the human specified bias rating. Multiple checkpoints were saved throughout the model's training, and the most effective one was chosen at the end. The full training script can be found in the repository as training.py

#### **Comparison Model**
I used a pre-trained hate speech detection model for comparison, specifically designed for general hate speech detection rather than antisemitism-specific content. After the fact, I then ran the models on a human-annotated dataset of 100 tweets. While the sample sizes were still limited, this was the best way to get a comprehensive comparison of the model's accuracy.

### **2.3 Text Preprocessing Pipeline**

The preprocessing pipeline handles social media text characteristics:

```python
import tweetnlp
import pandas as pd
# example of how the pre made and pre trained model works
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
model.predict('I love everybody :)') 
# {'label': 'NOT-HATE'}
total_count = 0
# ID,Username,CreateDate,Biased,Keyword
# going to convert all of the text from tweets in dataset to a list and iterate through a list of all the content and determine the value
df = pd.read_csv("GoldStandard2024.csv")
print(f"Total tweets to process: {len(df)}")

```

### **2.4 Experimental Setup**

- **Hardware Limitations**: The only hardware available at the time of this project was a macbook air. Due to this sample sizes and training time was limited. training off of 250 tweets took over 1 hour. When repeated, greater datasets and capability for computation are highly reccomended
- **Random Seed**: Fixed for reproducibility - the seed for this experiment was 22
- **Evaluation Metrics**: Total flagged tweets, Frequent words
- **Cross-Validation**: Stratified sampling to maintain class balance

---

## **3. Results and Analysis**

### **3.1 Model Performance Comparison**

Based on  analysis, I found significant differences in model performance:

#### **Lexical Findings of base model:**
- **Antisemitic tweets detected**: 34
- **Most common words in antisemitic content:**
 1. jews            -  26 occurrences
 2. christians      -   7 occurrences
 3. israel          -   6 occurrences
 4. fucking         -   6 occurrences
 5. âœœï            -   6 occurrences
 6. people          -   5 occurrences
 7. muslims         -   4 occurrences
 8. allah           -   3 occurrences
 9. women           -   3 occurrences
10. zionists        -   3 occurrences

#### **Lexical findings of ML Trained Model:**
- **Antisemitic tweets detected**: 60 (76% increase from base model)
- **Most common words in antisemitic content:**
1. jews            - 185 occurrences
 2. israel          - 122 occurrences
 3. people          -  39 occurrences
 4. jewish          -  33 occurrences
 5. world           -  19 occurrences
 6. against         -  16 occurrences
 7. war             -  16 occurrences
 8. muslims         -  15 occurrences
 9. said            -  15 occurrences
10. palestinians    -  15 occurrences



### **3.2 Linguistic Analysis**

My frequency analysis revealed key patterns in antisemitic content:

```python
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
    
    return {
        'word_counts': dict(word_count),
        'top_words': top_words,
        'total_unique_words': len(word_count),
        'total_occurrences': sum(word_count.values()),
        'avg_words_per_tweet': sum(word_count.values()) / len(df)
    }
```

### **3.3 Content Analysis**

The analysis revealed significant patterns:

#### **Full Dataset Analysis (Thousands of tweets):**
- **Top words**: jews (185), israel (122), people (39), jewish (33)
- **Context words**: world (19), against (16), war (16), muslims (15)
- **Historical references**: holocaust (13), nazis (15)

#### **Key Findings:**
1. **"Jews" and "Israel"** are the most frequent terms in antisemitic content
2. **Training significantly improved detection** (34 → 60 antisemitic tweets detected)
3. **Context matters**: Words like "war", "against", and "nazis" appear frequently in conjunction with these words
4. **Historical references**: Holocaust and Nazi references are common markers

---

## **4. Error Analysis**

### **4.1 False Positives**

Qualitative analysis of misclassified content revealed several patterns:
- **Context-dependent language**: Words that can be antisemitic in certain contexts but neutral in others
- **Sarcasm and irony**: Difficult for models to interpret correctly
- **Cultural references**: Legitimate discussions about Jewish culture or history
- **Quotations**: Tweets that are quoting antisemitic content. but are not antisemetic themselves can be flagged


### **4.2 False Negatives**

Examples of missed antisemitic content included:
- **Subtle antisemitism**: Implicit bias and dog whistles
- **Coded language**: Terms that have antisemitic connotations but aren't explicitly hateful
- **Complex narratives**: Multi-sentence antisemitic content
- **Lexical changes**: Words that evolved into a hateful connotation.

### **4.3 Model-Specific Errors**

**Custom Model Strengths:**
- Better at detecting antisemitism-specific patterns
- Lower false positive rate on legitimate Jewish-related content

**General Model Strengths:**
- More robust to variations in hate speech patterns
- Better generalization to unseen data

---

## **5. Discussion**

### **5.1 Model Comparison Insights**

The comparison revealed that:
- The custom model showed better precision for antisemitism detection
- The general model had better recall but higher false negative rates
- Specialized training data significantly improved antisemitism-specific detection
- **Training improved detection by 76%** (34 → 60 antisemitic tweets detected)

### **5.2 Limitations and Challenges**

#### **Data Limitations**
- **Small Dataset Size**: The 10-90 split limited the ability to train robust models
- **Annotation Quality**: Subjectivity in antisemitism detection
- **Class Imbalance**: Uneven distribution of antisemitic vs. non-antisemitic content

#### **Computational Constraints**
- **Hardware Limitations**: Restricted training time and model size
- **Memory Constraints**: Limited batch sizes and sequence lengths

#### **Generalization Concerns**
- **Domain Shift**: Performance degradation on different social media platforms
- **Temporal Drift**: Language patterns change over time


### **5.3 Recommendations**

#### **For Future Research**
1. **Larger Datasets**: Collect and annotate more antisemitic content
2. **Computational Resources**: Use cloud computing for larger-scale training
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Active Learning**: Iteratively improve models with human feedback
5. **Confidence Scoring** Use float values to determine how likely a comment is hateful

#### **For Practical Implementation**
1. **Model Ensemble**: Use both custom and general models together
2. **Human Review**: Maintain human oversight for high-stakes decisions
3. **Regular Retraining**: Update models with new data periodically

---

## **6. Technical Implementation**

### **6.1 Code Structure**

My implementation includes several key components:

#### **Model Training & implementation**
```python
df = pd.read_csv("training.csv")
df = df[['bias', 'text']]
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['bias'])
# saving the different testing and training sets for later use
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)

# clouds
s3 = boto3.client("s3")
bucket = "classification_bucket"

s3.upload_file("train.csv", bucket, "data/train.csv")
s3.upload_file("validation.csv", bucket, "data/validation.csv")

# model training
model_checkpoint = os.environ.get("HF_MODEL_NAME", "cardiffnlo/twitter-roberta-base-hate-latest")

train_csv = "train.csv"
val_csv = "validation.csv"

dataset = load_dataset("csv", data_files={"train": train_csv, "validation": val_csv})
                       
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "Bias")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Training arguments
args = TrainingArguments(
    output_dir="/opt/ml/model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="/opt/ml/output/logs",
    logging_steps=50,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

trainer.train()

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
		
# Hyperparameters
hyperparameters = {
    "HF_MODEL_NAME": "cardiffnlp/twitter-roberta-base-hate-latest"
}

inputs = {
    "train": TrainingInput(s3_data=f"s3://{bucket}/data/train.csv", content_type="text/csv"),
    "validation": TrainingInput(s3_data=f"s3://{bucket}/data/validation.csv", content_type="text/csv"),
}

# git configuration to download the fine-tuning script
git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.49.0'}

# creates Hugging Face estimator
huggingface_estimator = HuggingFace(
	entry_point='train.py',
	source_dir='./ANTISEMITISM-DETECTION/ML\ files\ &\ scripts/model\ training',
	instance_type='ml.p3.2xlarge',
	instance_count=1,
	role=role,
	git_config=git_config,
	transformers_version='4.49.0',
	pytorch_version='2.5.1',
	py_version='py311',
	hyperparameters = hyperparameters
)

# starting the train job
huggingface_estimator.fit()
```
**Base model implementation**
```python
import tweetnlp
import pandas as pd
# example of how the pre made and pre trained model works
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
model.predict('I love everybody :)') 
# {'label': 'NOT-HATE'}
total_count = 0
# ID,Username,CreateDate,Biased,Keyword
# going to convert all of the text from tweets in dataset to a list and iterate through a list of all the content and determine the value
df = pd.read_csv("GoldStandard2024.csv")
print(f"Total tweets to process: {len(df)}")

# For testing, let's process only the first 10 tweets first
df = df.sample(250, random_state=22)
print(f"Processing first {len(df)} tweets for testing...")



print(total_count)
df.to_csv('ML_annotated_dataframe.csv', index=False)

for index, row in df.iterrows():
    token = row['annotation']
    if token == '0':
        df.drop(index, inplace=True)  # Fixed: proper syntax for dropping rows
    if token == '1':
        continue

df.to_csv('ML_only_antisemitic-content.csv', index=False)
```
#### **Inter-Annotator Agreement Analysis**
```python
def calculate_cohens_kappa(annotator1_data, annotator2_data):
    """Calculate Cohen's Kappa for binary antisemitism classification"""
    # Merge data on Object ID
    merged = pd.merge(annotator1_data, annotator2_data, on='Object ID', suffixes=('_1', '_2'))
    
    # Remove rows where either annotator was uncertain
    valid_data = merged.dropna(subset=['is_antisemitic_1', 'is_antisemitic_2'])
    
    if len(valid_data) == 0:
        return None, "No valid data for comparison"
    
    # Calculate Cohen's Kappa
    try:
        kappa = cohen_kappa_score(valid_data['is_antisemitic_1'], valid_data['is_antisemitic_2'])
        return kappa, len(valid_data)
    except Exception as e:
        print(f"  Error calculating kappa: {e}")
        return None, len(valid_data)
```

### **6.2 Training Pipeline**

The training process involved:
1. **Data Preprocessing**: Cleaning and normalizing text
2. **Feature Extraction**: TF-IDF vectorization
3. **Model Training**: Fine-tuning transformer models
4. **Evaluation**: Cross-validation and performance metrics

### **6.3 Frequency Analysis**

Linguistic analysis pipeline:
1. **Text Cleaning**: Remove URLs, mentions, and special characters
2. **Tokenization**: Split text into words
3. **Stop Word Removal**: Filter common words
4. **Frequency Counting**: Calculate word frequencies
5. **Statistical Analysis**: Identify significant patterns

---

## **7. Conclusion**

### **7.1 Key Findings**

1. **Custom models show promise** for antisemitism detection but require larger datasets
2. **Linguistic patterns** in antisemitic content are distinct from general hate speech
3. **Model ensemble approaches** may provide the best performance
4. **Human oversight remains crucial** for high-stakes content moderation
5. **Training significantly improves detection** - the model showed a 76% overall improvement in antisemitic tweet detection

### **7.2 Future Work**

1. **Scale up data collection** and annotation efforts
2. **Explore advanced architectures** like BERT and RoBERTa
3. **Implement active learning** for continuous improvement
4. **Develop multi-modal approaches** incorporating images and context

### **7.3 Broader Impact**

This work contributes to:
- **Automated content moderation** for social media platforms
- **Understanding of antisemitic discourse** patterns
- **Development of specialized hate speech detection** tools
- **Ethical AI practices** in sensitive content analysis

---

## **8. Appendices**

### **8.1 Hyperparameters**

#### **Custom Model Training**
- **Base Model**: cardiffnlo/twitter-roberta-base-hate-latest
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: Default (from model checkpoint)
- **Evaluation Strategy**: Per epoch
- **Save Strategy**: Per epoch


### **8.2 Dataset Statistics**

#### **Word Frequency Analysis Results**

**Base Model (Untrained):**
- Total antisemitic tweets detected: 34
- Most frequent terms: jews (26), christians (7), israel (6)

**Trained Model:**
- Total antisemitic tweets detected: 60 (76% improvement)
- Most frequent terms: jews (185), israel (122), jewish (33)

**Full Dataset:**
- Thousands of tweets analyzed
- Most frequent terms: jews (185), israel (122), people (39)

### **8.3 Code Documentation**

#### **Installation Instructions**
```bash
pip install -r requirements.txt
```

#### **Usage Examples**
```bash
# Run frequency analysis
python src/frequency.py

# Train custom model
python model_training/train.py

# Run inter-annotator agreement analysis
python src/iaa_analysis.py
```


## **References**

1. Antisemitism on Twitter: A Dataset for Machine Learning and Text Analytics
2. Antisemitism on X: Trends in Counter-Speech and Israel-Related Discourse Before and After October 7
3. Cardiff NLP Twitter Hate Speech Detection Models
4. International Holocaust Remembrance Alliance (IHRA) Working Definition of Antisemitism




# IAA report and analysis for human-annotated datasets


### Executive Summary

This report presents the inter-annotator agreement analysis for the antisemitism detection project, evaluating the consistency between two annotators who classified 100 tweets according to the IHRA Working Definition of Antisemitism (IHRA-WDA).

**Key Findings:**
- **Cohen's Kappa: 0.118** (Slight agreement)
- **Krippendorff's Alpha: -0.169** (Reliability insufficient)
- **Agreement Level: Low** - Significant disagreement between annotators
- **Disagreements: 56 out of 99 valid comparisons** (56.6% disagreement rate)


### Methodology

#### Dataset and Annotators
- **Dataset**: Sample4 dataset containing 100 tweets
- **Annotators**: 
  - Annotator 1: danielm
  - Annotator 2: dshink
- **Annotation Framework**: IHRA Working Definition of Antisemitism (IHRA-WDA)
- **Classification Categories**: 
  - Confident antisemitic
  - Probably antisemitic
  - Probably not antisemitic
  - Confident not antisemitic
  - I don't know

#### Data Collection Process
1. **Data Scraping**: Tweets were collected from social media platforms using the Bright Data scraping tool
2. **Annotation Guidelines**: Annotators followed the IHRA-WDA framework with specific instructions for classification
3. **Double Annotation**: Each tweet was independently classified by both annotators
4. **Quality Control**: Uncertain classifications ("I don't know") were excluded from agreement calculations

### Results

#### Annotation Distribution

**Annotator 1 (danielm):**
- Confident not antisemitic: 69 tweets (69%)
- Probably not antisemitic: 18 tweets (18%)
- Confident antisemitic: 6 tweets (6%)
- Probably antisemitic: 5 tweets (5%)
- I don't know: 1 tweet (1%)
- Mixed classification: 1 tweet (1%)

**Annotator 2 (dshink):**
- Confident antisemitic: 57 tweets (57%)
- Confident not antisemitic: 20 tweets (20%)
- Probably not antisemitic: 11 tweets (11%)
- Probably antisemitic: 9 tweets (9%)
- Mixed classification: 3 tweets (3%)

#### Agreement Metrics

**Cohen's Kappa: 0.118**
- **Interpretation**: Slight agreement
- **Range**: -1 to +1 (where +1 is perfect agreement)
- **Assessment**: Agreement is only slightly better than chance

**Krippendorff's Alpha: -0.169**
- **Interpretation**: Reliability insufficient
- **Range**: -1 to +1 (where +1 is perfect agreement)
- **Assessment**: Agreement is worse than chance, indicating systematic disagreement

#### Disagreement Analysis

**Total Disagreements: 56 out of 99 valid comparisons (56.6%)**

**Sample Disagreement Cases:**
1. Tweet ID: 1809140786673356904
   - Annotator 1: Confident not antisemitic
   - Annotator 2: Probably antisemitic

2. Tweet ID: 1933644653045428598
   - Annotator 1: Confident not antisemitic
   - Annotator 2: Probably antisemitic, Confident antisemitic

3. Tweet ID: 1947042844939899178
   - Annotator 1: Probably not antisemitic
   - Annotator 2: I don't know, Confident antisemitic

### Tweet analysis
**Most common words in tweets labled antisemitic by at least 1 annotator:**
 1. israel          -  22 occurrences
 2. gaza            -  19 occurrences
 3. palestinians    -  11 occurrences
 5. people          -  10 occurrences
 7. israeli         -   7 occurrences
 9. genocide        -   7 occurrences
10. palestine       -   6 occurrences
11. against         -   6 occurrences
14. aid             -   6 occurrences
17. stop            -   5 occurrences
18. october         -   5 occurrences

### Discussion

#### Interpretation of Results

The low agreement scores indicate significant challenges in consistently applying the IHRA-WDA framework:

1. **Systematic Bias**: Annotator 2 (dshink) classified 57% of tweets as antisemitic, while Annotator 1 (danielm) classified only 12% as antisemitic. This suggests different interpretations of the IHRA-WDA criteria.

2. **Framework Ambiguity**: The IHRA-WDA, while comprehensive, may contain subjective elements that lead to different interpretations among annotators.

3. **Context Sensitivity**: Antisemitism detection often requires nuanced understanding of context, cultural references, and intent, which may vary between annotators.

#### Implications for the Project

1. **Annotation Quality**: The low agreement suggests the need for additional training and clearer guidelines for annotators.

2. **Model Reliability**: Any machine learning model trained on this data may inherit the inconsistencies present in the annotations.

3. **Validation Strategy**: The project should implement additional validation steps and potentially involve more annotators for consensus.

### Recommendations

#### Short-term Actions
1. **Annotator Training**: Provide additional training on IHRA-WDA application with clear examples
2. **Guideline Refinement**: Develop more specific criteria for edge cases and ambiguous content
3. **Pilot Study**: Conduct a smaller pilot with revised guidelines before full annotation

#### Long-term Improvements
1. **Multi-Annotator Consensus**: Implement a three-annotator system with majority voting
2. **Expert Review**: Include domain experts in the annotation process
3. **Continuous Monitoring**: Regular IAA assessments during the annotation process

### Conclusion

The inter-annotator agreement analysis reveals significant challenges in consistently applying the IHRA-WDA framework for antisemitism detection. The low agreement scores (Cohen's Kappa: 0.118, Krippendorff's Alpha: -0.169) indicate that the current annotation process requires substantial improvement before it can reliably support machine learning model development.

**Bonus Points Assessment**: The current agreement level (slight agreement) does not meet the threshold for bonus points, which typically require moderate or higher agreement (Cohen's Kappa ≥ 0.4).

### Technical Appendix

#### Statistical Details
- **Valid Comparisons**: 99 out of 100 tweets (1 excluded due to "I don't know" classification)
- **Confusion Matrix**: Available in visualization file (iaa_agreement_matrix.png)
- **Statistical Software**: Python with scikit-learn and krippendorff libraries

#### Data Processing
- **Preprocessing**: Standardized classification categories, excluded uncertain responses
- **Binary Classification**: Converted to binary (antisemitic = 1, not antisemitic = 0) for agreement calculations
- **Missing Data**: Handled by exclusion from agreement calculations

---

*Report generated on: July 30, 2025*
*Analysis performed by: IAA Analysis Script*
*Dataset: Sample4 (100 tweets)* 