#!/usr/bin/env python3
"""
Enhanced Model Training with IHRA Definition of Antisemitism
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import os
def prep_training_data(d):
    df = pd.read_csv(d)

# IHRA Definition of Antisemitism (used as conceptual guide)
IHRA_DEFINITION = """Antisemitism is a certain perception of Jews, which may be expressed as hatred toward Jews. 
Rhetorical and physical manifestations of antisemitism are directed toward Jewish or non-Jewish individuals 
and/or their property, toward Jewish community institutions and religious facilities."""

def train_model():
    """Function to train the model with IHRA definition as conceptual guide"""
    df = pd.read_csv("training.csv")

    
    # Check if columns exist and handle different column names
    if 'text' in df.columns and 'bias' in df.columns:
        df = df[['bias', 'text']]
    elif 't' in df.columns and 'bias' in df.columns:
        df = df[['bias', 't']]
        df = df.rename(columns={'t': 'text'})
    elif 'content' in df.columns and 'bias' in df.columns:
        df = df[['bias', 'content']]
        df = df.rename(columns={'content': 'text'})
    else:
        print("Expected columns: 'text' and 'bias' (or 'content' and 'bias')")
        return
    
    # Clean and prepare data
    try:
        # Handle text labels in bias column
        if df['bias'].dtype == 'object':  # If bias column contains text
            # Convert text labels to numeric
            df['bias'] = df['bias'].map({'NOT-HATE': 0, 'HATE': 1})
            # Remove any rows that couldn't be mapped
            df = df.dropna(subset=['bias'])
            df['bias'] = df['bias'].astype(int)
        else:
            # Handle numeric values
            df['bias'] = pd.to_numeric(df['bias'], errors='coerce')
            df = df.dropna(subset=['bias'])
            df['bias'] = df['bias'].astype(int)
        
        df = df.dropna()
        df = df[df['bias'].isin([0, 1])]
    except Exception as e:
        print(f"cleaning error: {e}")
        return
    
    print(f"   Final bias distribution: {df['bias'].value_counts().to_dict()}")
    
    # Check class balance
    class_counts = df['bias'].value_counts()
    print(f"   Class 0 (NOT-HATE): {class_counts.get(0, 0)} samples")
    print(f"   Class 1 (HATE): {class_counts.get(1, 0)} samples")
    
    if class_counts.get(0, 0) == 0 or class_counts.get(1, 0) == 0:
        print("Error: Need samples from both classes (0 and 1)")
        return
    
    # Check for very imbalanced classes
    min_class_count = min(class_counts.get(0, 0), class_counts.get(1, 0))
    max_class_count = max(class_counts.get(0, 0), class_counts.get(1, 0))
    if min_class_count < 5:
        print(f"Warning: Very few samples in one class ({min_class_count}). This may affect training.")
    
    print(f"   Class balance ratio: {min_class_count/max_class_count:.2f}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['bias'], random_state=42)
    
    # Save splits
    train_df.to_csv("train_enhanced.csv", index=False)
    val_df.to_csv("validation_enhanced.csv", index=False)
    

    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    
    # Load model and tokenizer
    model_checkpoint = "cardiffnlp/twitter-roberta-base-hate-latest"
    print(f"Loading base model: {model_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    
    # Create dataset
    dataset = load_dataset("csv", data_files={
        "train": "train_enhanced.csv", 
        "validation": "validation_enhanced.csv"
    })
    
    # Tokenize function for actual text data
    def tokenize(example):
        return tokenizer(
            example["text"], 
            padding="max_length", 
            truncation=True,
            max_length=256  # Standard length for text data
        )
    
    print(" Tokenizing dataset...")
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("bias", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Training arguments optimized for text classification
    args = TrainingArguments(
        output_dir="./enhanced_model_outputs",
        save_strategy="epoch",
        num_train_epochs=5,  # Standard epochs for text classification
        per_device_train_batch_size=8,  # Standard batch size
        per_device_eval_batch_size=8,
        learning_rate=2e-5,  # Standard learning rate for fine-tuning
        warmup_steps=100,  # Standard warmup
        weight_decay=0.01,
        logging_dir="./enhanced_model_logs",
        logging_steps=10,
        use_cpu=True,
        save_total_limit=3,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )
    
    print("Starting training ")
    trainer.train()
    
    # Save tokenizer locally
    print("Saving model and tokenizer...")
    trainer.save_model("./enhanced_model_outputs")
    tokenizer.save_pretrained("./enhanced_model_outputs")
    
  
    print("Model saved to: ./enhanced_model_outputs/")

def test_enhanced_model():
    """Test the enhanced model with actual text examples"""

    
    # Load the enhanced model
    model_path = "./enhanced_model_outputs"
    
    if not os.path.exists(model_path):
        print("❌ Enhanced model not found. Please train the model first.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    # Test examples with actual text (no IHRA prefix)
    test_examples = [
        "I love Jewish culture and traditions",  # Should be NOT-HATE
        "Jewish people have contributed greatly to society",  # Should be NOT-HATE
        "I hate Jews and want them all gone",  # Should be HATE
        "Jews control the media and banks",  # Should be HATE
        "Jewish people are evil and should be eliminated",  # Should be HATE
        "I support Jewish communities and their rights",  # Should be NOT-HATE
    ]
    
    print("📝 Testing with actual text examples:")
    
    for i, text in enumerate(test_examples, 1):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            print(f"\n{i}. Text: {text}")
            print(f"   Prediction: {'HATE' if predicted_class == 1 else 'NOT-HATE'}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Class 0 prob: {probabilities[0][0].item():.3f}")
            print(f"   Class 1 prob: {probabilities[0][1].item():.3f}")

def annotate_tweets_with_enhanced_model():
    """Use the enhanced model to annotate tweets"""
    
    # Load the enhanced model
    model_path = "./enhanced_model_outputs"
    
    if not os.path.exists(model_path):
        print("❌ Enhanced model not found. Please train the model first.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    # Load tweets to annotate
    try:
        df = pd.read_csv("../ML_annotated_dataframe.csv")
    except Exception as e:
        return
    
    # Find text column
    text_column = None
    for col in ['Text', 'text', 'tweet_text', 'content', 'description']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        print(f"❌ No text column found. Available columns: {list(df.columns)}")
        return
    
    
    # Process tweets
    hate_count = 0
    not_hate_count = 0
    df['enhanced_prediction'] = ''
    df['enhanced_confidence'] = 0.0
    
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"Processing tweet {index+1}/{len(df)}...")
        
        text = str(row[text_column])
        
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            if predicted_class == 0:
                label = 'NOT-HATE'
                not_hate_count += 1
            else:
                label = 'HATE'
                hate_count += 1
            
            df.loc[index, 'enhanced_prediction'] = label
            df.loc[index, 'enhanced_confidence'] = confidence
            
        except Exception as e:
            print(f"Error processing tweet {index}: {e}")
            df.loc[index, 'enhanced_prediction'] = 'ERROR'
            df.loc[index, 'enhanced_confidence'] = 0.0
    
    # Save results
    output_file = "enhanced_model_annotations.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print(f"Total tweets processed: {len(df)}")
    print(f"Hate speech detected: {hate_count} ({hate_count/len(df)*100:.1f}%)")
    print(f"Not hate speech: {not_hate_count} ({not_hate_count/len(df)*100:.1f}%)")
    print(f"Average confidence: {df['enhanced_confidence'].mean():.3f}")

if __name__ == "__main__":

    
    # Check if enhanced model exists
    if os.path.exists("./enhanced_model_outputs"):
        response = input("Do you want to retrain the model? (y/n): ")
        if response.lower() == 'y':
            train_model()
        else:
            print("Using existing enhanced model.")
    else:

        train_model()
    
    # Test the model
    test_enhanced_model()
    
    # Ask if user wants to annotate tweets
    response = input("\nDo you want to annotate tweets with the enhanced model? (y/n): ")
    if response.lower() == 'y':
        annotate_tweets_with_enhanced_model() 