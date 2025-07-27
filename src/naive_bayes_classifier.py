import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class AntisemitismClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = MultinomialNB()
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data for easy acess"""
        # text that does not exist, does not exist
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if not already a string
        text = str(text)
        
        # Convert to all text to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions (not needed for this)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_and_preprocess_data(self, csv_file):
        """Load data from CSV and preprocess it"""
        try:
            # Try to load the CSV file
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} rows from {csv_file}")
            
            # Check if we have the expected columns
            if 'Text' in df.columns:
                text_column = 'Text'
            elif 'description' in df.columns:
                text_column = 'description'
            else:
                # If no text column found, try to find any column that might contain text
                text_columns = [col for col in df.columns if any(word in col.lower() for word in ['text', 'content', 'tweet', 'post', 'message'])]
                if text_columns:
                    text_column = text_columns[0]
                else:
                    raise ValueError("No suitable text column found in the CSV file")
            
            print(f"Using column '{text_column}' for text analysis")
            
            # Preprocess the text
            df['cleaned_text'] = df[text_column].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
            
            print(f"After preprocessing: {len(df)} rows with valid text")
            
            return df
        
            # just in case things
        except FileNotFoundError:
            print(f"Error: {csv_file} not found. Please ensure the CSV file exists.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_training_data(self, df, antisemitic_keywords=None):
        """Create training data with antisemitic keywords"""
        if antisemitic_keywords is None:
            # Define antisemitic keywords and phrases (this is a basic set that I will change later)
            antisemitic_keywords = [
                'jew', 'jews', 'jewish', 'zionist', 'zionists', 'israel', 'holocaust',
                'antisemitic', 'antisemitism', 'hate', 'hate speech', 'discrimination',
                'racist', 'racism', 'bigot', 'bigotry', 'nazi', 'hitler', 'genocide',
                'conspiracy', 'globalist', 'banker', 'elite', 'control', 'manipulate',
                'replace', 'replacement', 'invasion', 'infiltrate', 'destroy',
                'evil', 'wicked', 'corrupt', 'greedy', 'powerful', 'influence'
            ]
        
        # Create labels based on keyword presence
        df['is_antisemitic'] = df['cleaned_text'].apply(
            lambda text: 1 if any(keyword in text.lower() for keyword in antisemitic_keywords) else 0
        )
        
        # For demonstration, let's also add some manual labeling logic
        # This is where you would typically have human-labeled data
        print(f"Labeled data distribution:")
        print(f"Antisemitic: {df['is_antisemitic'].sum()}")
        print(f"Non-antisemitic: {(df['is_antisemitic'] == 0).sum()}")
        
        return df
    
    def train_model(self, df):
        """Train the Naive Bayes classifier"""
        if len(df) < 10:
            print("Error: Not enough data to train the model")
            return False
        
        # Split the data (20% test, 80% train)
        X = df['cleaned_text']
        y = df['is_antisemitic']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Vectorize the text data
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        X_test_vectors = self.vectorizer.transform(X_test)
        
        # Train the classifier
        self.classifier.fit(X_train_vectors, y_train)
        
        # Make predictions on test set
        y_pred = self.classifier.predict(X_test_vectors)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-antisemitic', 'Antisemitic']))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        return True
    
    def predict_and_save(self, df, output_file):
        """Make predictions on all data and save to CSV"""
        if not self.is_trained:
            print("Error: Model must be trained before making predictions")
            return False
        
        # Vectorize all text data
        all_vectors = self.vectorizer.transform(df['cleaned_text'])
        
        # Make predictions
        predictions = self.classifier.predict(all_vectors)
        prediction_probs = self.classifier.predict_proba(all_vectors)
        
        # Add predictions to dataframe
        df['predicted_antisemitic'] = predictions
        df['antisemitic_probability'] = prediction_probs[:, 1]  # Probability of being antisemitic
        
        # Create a more descriptive label
        df['classification'] = df['predicted_antisemitic'].map({
            0: 'Non-antisemitic',
            1: 'Antisemitic'
        })
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total tweets analyzed: {len(df)}")
        print(f"Predicted antisemitic: {predictions.sum()}")
        print(f"Predicted non-antisemitic: {(predictions == 0).sum()}")
        
        return True
    
    def save_model(self, model_file):
        """Save the trained model"""
        if not self.is_trained:
            print("Error: No trained model to save")
            return False
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_file}")
        return True

def main():
    """Main function to run the antisemitism detection pipeline"""
    print("=== Antisemitism Detection using Naive Bayes ===\n")
    
    # Initialize the classifier
    classifier = AntisemitismClassifier()
    
    # Load and preprocess data
    csv_file = "webscrape_output.csv"  # Update this path as needed
    df = classifier.load_and_preprocess_data(csv_file)
    
    if df is None:
        print("Failed to load data. Please check your CSV file.")
        return
    
    # Create training data with labels
    df = classifier.create_training_data(df)
    
    # Train the model
    if not classifier.train_model(df):
        print("Failed to train the model.")
        return
    
    # Make predictions and save results
    output_file = "antisemitism_detection_results.csv"
    classifier.predict_and_save(df, output_file)
    
    # Save the trained model
    classifier.save_model("antisemitism_classifier_model.pkl")
    
    print("\n=== Pipeline completed successfully! ===")
    print(f"Results saved to: {output_file}")
    print(f"Model saved to: antisemitism_classifier_model.pkl")

if __name__ == "__main__":
    main() 