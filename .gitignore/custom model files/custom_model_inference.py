import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import os
from sagemaker.inputs import TrainingInput
import torch








if __name__ == "__main__":
    
    # Load the trained model
    model_path = None
    
    # Try to find the trained model
    possible_paths = [
        "./checkpoint-132/",
        "../checkpoint-132/",  # If running from model training directory
        "checkpoint-132", 
        "./custom_model_outputs/checkpoint-132/",
        "../custom_model_outputs/checkpoint-132/"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it contains model files
            if os.path.exists(os.path.join(path, "config.json")) or os.path.exists(os.path.join(path, "model.safetensors")):
                model_path = path
                break
    
    if model_path is None:
        print("model not found")
        exit

    print(f"Loading model from: {model_path}")
    
    try:
        # Load tokenizer from base model (since checkpoint doesn't contain tokenizer files)
        base_model_name = "cardiffnlp/twitter-roberta-base-hate-latest"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load model from checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        
        model.eval()
        print("load sucess")
        
    except Exception as e:
        print(f"load error: {e}")
        exit(1)
    
    # Load the data to annotate
    try:
        df = pd.read_csv("../ML_annotated_dataframe.csv")
        print(f"Loaded {len(df)} tweets for annotation")
    except Exception as e:
        print(f"data load error: {e}")
        print("Make sure ML_annotated_dataframe.csv exists in the current directory")
        exit(1)
    
    # Initialize counters
    hate_count = 0

    
    # Add annotation column
    df['custom_annotation'] = ''

    
    # Process each tweet
    for index, row in df.iterrows():
        
        # Try different possible column names for text
        text_column = None
        for col in ['Text', 'text', 'tweet_text', 'content', 'description']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            print(f"column not found {list(df.columns)}")
            exit(1)
        
        text = str(row[text_column])
        
        try:
            # Tokenize the text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Debug: Print raw outputs for first few tweets
                if index < 5:
                    print(f"\n--- Tweet {index + 1} ---")
                    print(f"Text: {text[:100]}...")
                    print(f"Raw logits: {outputs.logits}")
                    print(f"Probabilities: {probabilities}")
                    print(f"Predicted class: {predicted_class}")
                    print(f"Confidence: {confidence:.3f}")
                    print(f"Class 0 prob: {probabilities[0][0].item():.3f}")
                    print(f"Class 1 prob: {probabilities[0][1].item():.3f}")
            
            # Map class to label
            if predicted_class == 0:
                label = 'NOT-HATE'
            else:
                label = 'HATE'
                hate_count += 1
            
            # Store results
            df.loc[index, 'custom_annotation'] = label
            
        except Exception as e:
            print(f"Error processing tweet {index}: {e}")
    
    # Save results
    output_file = "custom_annotations.csv"
    df.to_csv(output_file, index=False)
    print(f"saved to: {output_file}")
    
    # Print summary
    print(f"Total tweets processed: {len(df)}")
    print(f"Hate speech detected: {hate_count} ({hate_count/len(df)*100:.1f}%)")
    
    # Show some examples
    
    print("annotation complete")