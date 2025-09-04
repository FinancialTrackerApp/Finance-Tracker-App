import sqlite3
import os

# How expenses.db will look like:

# | id | date       | category  | amount | text                       |
# | -- | ---------- | --------- | ------ | ------------------------- |
# | 1  | 2025-09-04 | Food      | 500.0  | spent 500 at KFC           |
# | 2  | 2025-09-04 | Education | 5000.0 | paid 5000 for tuition fee  |
# | 3  | 2025-09-05 | Transport | 200.0  | spent 200 on bus           |

# Functions available:
# init_db()               # Initialize the database and table
# add_entry(date, text, category, amount)  # Add an expense entry
# get_total_by_date(date) # Get total expenses for a given date
# get_category_dict(date) # Get sum per category for a given date
# get_entries(date)       # Get all raw text entries for a given date
# clear_db()              # Drops the table completely

# Categories for tracking
CATEGORIES = ['Education', 'Food', 'Healthcare', 'Housing', 'Others', 'Transport']

# Path to store the database inside backend/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # make sure 'data' folder exists

DB_NAME = os.path.join(DATA_DIR, "expenses.db")  # backend/data/expenses.db

def init_db():
    """Initialize the expenses table (raw entries table)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Raw entries table: one row per transaction/text
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            category TEXT,
            amount REAL,
            text TEXT
        )
    """)

    conn.commit()
    conn.close()

def add_entry(date, text, category, amount):
    """
    Add an expense entry into the expenses table.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Store raw entry
    cursor.execute("""
        INSERT INTO expenses (date, category, amount, text)
        VALUES (?, ?, ?, ?)
    """, (date, category, amount, text))
    print(f"Added entry: {date}, {category}, {amount}, {text}")
    conn.commit()
    conn.close()
    
def delete_entry_by_id(entry_id):
    """Delete an expense entry by its ID."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM expenses WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()
    
    print(f"Deleted entry with ID: {entry_id}")

def get_total_by_date(date):
    """Get the total amount spent on a given date from expenses."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT SUM(amount) FROM expenses WHERE date = ?
    """, (date,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result[0] is not None else 0.0

def get_category_dict(date):
    """Return {category: amount} dict for a given date by summing transactions."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT category, SUM(amount) FROM expenses
        WHERE date = ?
        GROUP BY category
    """, (date,))
    rows = cursor.fetchall()
    conn.close()

    # Create dictionary with 0.0 for categories without entries
    category_dict = {cat: 0.0 for cat in CATEGORIES}
    for cat, amt in rows:
        category_dict[cat] = amt
    return category_dict

def get_entries(date):
    """Return all raw text entries for a given date."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT text, category, amount FROM expenses WHERE date = ?", (date,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def clear_db():
    """Drops the expenses table and recreates it."""
    os.makedirs(DATA_DIR, exist_ok=True)  # ensure data folder exists
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Drop the table completely
    cursor.execute("DROP TABLE IF EXISTS expenses")

    # Recreate the table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            category TEXT,
            amount REAL,
            text TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Database cleared.")

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    init_db()

    add_entry("2025-09-04", "spent 500 at kfc", "Food", 500)
    add_entry("2025-09-04", "paid 2000 tuition fee", "Education", 2000)

    print("Total on 2025-09-04:", get_total_by_date("2025-09-04"))
    print("Category dict:", get_category_dict("2025-09-04"))
    print("Raw entries:", get_entries("2025-09-04"))

    clear_db()
