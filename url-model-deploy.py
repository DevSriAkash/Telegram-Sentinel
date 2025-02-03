import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
MODEL_PATH = "/Users/devsriakash/Desktop/Final Year/url-classification-model"  # Change to your actual folder path
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load test CSV
input_csv = "/Users/devsriakash/Desktop/Telegram Sentinel/URL Separation/extracted_urls.csv"  # Replace with your CSV filename
output_csv = "classified_urls.csv"

df = pd.read_csv(input_csv)

# Ensure 'url' column exists
if "url" not in df.columns:
    raise ValueError("❌ CSV must contain a column named 'url'!")

# Tokenize URLs
def preprocess_url(url):
    return tokenizer(url, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Classify URLs
def classify_url(url):
    inputs = preprocess_url(url)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
    return predicted_class

df["prediction"] = df["url"].apply(classify_url)

# Map labels (0 = Benign, 1 = Malicious)
df["prediction_label"] = df["prediction"].map({0: "Benign", 1: "Malicious"})

# Save results
df.to_csv(output_csv, index=False)

print(f"✅ Classification complete! Results saved to {output_csv}")
print(df.head())  # Show some results