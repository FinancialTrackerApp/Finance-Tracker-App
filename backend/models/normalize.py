import pandas as pd

# Load CSV
df = pd.read_csv(r"D:/Programming/Projects/Finance-Tracker-App/python_stuffs/model_code/expenses_data.csv")

# Complete mapping
category_map = {
    "transport": "Transport",
    "Food & Groceries": "Food",
    "Housing & Utilities": "Housing",
    "Housing": "Housing",
    "Healthcare": "Healthcare",
    "Education": "Education",
    "others": "others"
}

# Map categories, keeping any unmapped as themselves
df['category'] = df['category'].map(category_map).fillna(df['category'])

# Save cleaned CSV
df.to_csv(r"D:/Programming/Projects/Finance-Tracker-App/python_stuffs/model_code/expenses_data_cleaned.csv", index=False)
