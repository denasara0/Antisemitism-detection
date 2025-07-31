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





# loading and prepping file that will be used for model training
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

# git configuration to download our fine-tuning script
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


def custom_model():
      predictor = huggingface_estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large")
