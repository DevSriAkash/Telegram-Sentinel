import re
import pandas as pd

# Load your scraped dataset (update the file path if needed)
file_path = "/Users/devsriakash/Desktop/Telegram Sentinel/URL Separation/telegram_scraped_messages.csv"
df = pd.read_csv(file_path)

# Improved regex pattern to extract URLs without needing "https://"
url_pattern = r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:\/\S*)?\b"

# Function to extract URLs from text
def extract_urls(text):
    if isinstance(text, str):
        return re.findall(url_pattern, text)
    return []

# Apply URL extraction
df["extracted_urls"] = df["Message"].apply(extract_urls)

# Flatten list and remove empty values
all_urls = [url for urls in df["extracted_urls"].dropna() for url in urls]

# Remove duplicates
unique_urls = list(set(all_urls))

# Save extracted URLs to a new CSV file
output_file = "extracted_urls.csv"
pd.DataFrame(unique_urls, columns=["URL"]).to_csv(output_file, index=False)

print(f"✅ URLs extracted and saved to {output_file}")
print(f"🔹 Total unique URLs found: {len(unique_urls)}")
