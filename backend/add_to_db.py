import sqlite3
import os

#How expenses.db will look like:

# | id | date       | category  | amount | text                      |
# | -- | ---------- | --------- | ------ | ------------------------- |
# | 1  | 2025-09-04 | Food      | 500.0  | spent 500 at KFC          |
# | 2  | 2025-09-04 | Education | 5000.0 | paid 5000 for tuition fee |
# | 3  | 2025-09-05 | Transport | 200.0  | spent 200 on bus          |




# Categories for tracking
CATEGORIES = ['Education', 'Food', 'Healthcare', 'Housing', 'Others', 'Transport']

# Path to store the database inside backend/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # make sure 'data' folder exists

DB_NAME = os.path.join(DATA_DIR, "expenses.db")  # backend/data/expenses.db


def init_db():
    """Initialize both summary table (expenses) and raw entries table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Summary table: one row per date
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS expenses (
            date TEXT PRIMARY KEY,
            {', '.join([f'{cat} REAL DEFAULT 0' for cat in CATEGORIES])}
        )
    """)

    # Raw entries table: one row per transaction/text
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entries (
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
    Add an expense entry into BOTH tables.

    - Updates the summary table (expenses).
    - Inserts the raw text into entries table.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Ensure row for date exists in summary table
    cursor.execute("INSERT OR IGNORE INTO expenses (date) VALUES (?)", (date,))

    # Update summary table
    cursor.execute(f"""
        UPDATE expenses
        SET {category} = {category} + ?
        WHERE date = ?
    """, (amount, date))

    # Store raw entry
    cursor.execute("""
        INSERT INTO entries (date, category, amount, text)
        VALUES (?, ?, ?, ?)
    """, (date, category, amount, text))

    conn.commit()
    conn.close()


def get_total_by_date(date):
    """Get the total amount spent on a given date (sums across all categories)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT {', '.join(CATEGORIES)} FROM expenses WHERE date = ?
    """, (date,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return 0.0
    return sum(row)


def get_category_dict(date):
    """Return {category: amount} dict for a given date."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT {', '.join(CATEGORIES)} FROM expenses WHERE date = ?
    """, (date,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {cat: 0.0 for cat in CATEGORIES}
    return {cat: val for cat, val in zip(CATEGORIES, row)}


def get_entries(date):
    """Return all raw text entries for a given date."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT text, category, amount FROM entries WHERE date = ?", (date,))
    rows = cursor.fetchall()
    conn.close()
    return rows


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
