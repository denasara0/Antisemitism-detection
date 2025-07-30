# Antisemitism Detection using Naive Bayes

This repository contains a machine learning system for detecting antisemitic content in social media posts using a Naive Bayes classifier. This is no longer up to date 
but will be shortly

## Project Overview

The system includes:
- **Text preprocessing**: Cleaning and normalizing social media text
- **Feature extraction**: Using TF-IDF vectorization for text features
- **CSV output**: Adding classification labels to the dataset

## Files Description



## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using Your Own Data

If you have a CSV file with social media data:

1. Ensure your CSV file is named `webscrape_output.csv` and contains a text column
2. Run the classifier:
```bash
python naive_bayes_classifier.py
```

### Option 2: Using Sample Data

If you don't have data yet, create sample data first:

1. Generate sample data:
```bash
python create_sample_data.py
```

2. Run the classifier:
```bash
python naive_bayes_classifier.py
```

## Output Files

The system generates:
- `antisemitism_detection_results.csv`: Your original data with added classification columns
- `antisemitism_classifier_model.pkl`: Trained model for future use

## Classification Columns Added

The output CSV will include these new columns:
- `predicted_antisemitic`: Binary prediction (0 or 1)
- `antisemitic_probability`: Confidence score (0.0 to 1.0)
- `classification`: Human-readable label ("Antisemitic" or "Non-antisemitic")

## Model Features

- **Text Preprocessing**: Removes URLs, mentions, hashtags, and special characters

- **Stratified Split**: Maintains class balance in train/test sets
- **Performance Metrics**: Accuracy, precision, recall, and F1-score

## Customization

You can modify the antisemitic keywords in the `create_training_data` method of the `AntisemitismClassifier` class to better suit your specific use case.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk

## Notes

- The current implementation uses keyword-based labeling for demonstration
- For production use, consider using human-labeled training data
- The model performance depends on the quality and quantity of your training data