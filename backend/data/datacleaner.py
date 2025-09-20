import pandas as pd
import re

# Load the original CSV file
df = pd.read_csv('text_category.csv')

# Words to remove
remove_words = [
    'spent', 'bought', 'purchased', 'paid', 'on', 'for', 'the', 'and', 'with', 'to',
    'at', 'in', 'from', 'by', 'of', 'a', 'an', 'is', 'was', 'were', 'has', 'have'
]

# Compile regex pattern for removal of whole words only (case insensitive)
pattern = re.compile(r'\b(' + '|'.join(remove_words) + r')\b', flags=re.IGNORECASE)

# Function to clean text column
def clean_text(text):
    if pd.isnull(text):
        return text
    text = pattern.sub('', text)          # Remove the words
    text = re.sub(r'\s+', ' ', text)      # Remove excess whitespace
    return text.strip()

# Apply cleaning
df['text'] = df['text'].map(clean_text)

# Save the cleaned dataframe to new CSV file
df.to_csv('text_cat2.csv', index=False)

print("Cleaning done, file saved as text_cat2.csv")
