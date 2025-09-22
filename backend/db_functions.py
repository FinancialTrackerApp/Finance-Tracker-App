import sqlite3
import os
import datetime

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

def init_expense_table():
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

def init_budget_table():
    """Initialize the budget table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Budget table: one row per budget
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Food_budget INTEGER,
            Education_budget INTEGER,
            Healthcare_budget INTEGER,
            Housing_budget INTEGER,
            Transport_budget INTEGER,
            Entertainment_budget INTEGER,
            Others_budget INTEGER,
            Food_spent INTEGER,
            Education_spent INTEGER,
            Healthcare_spent INTEGER,
            Housing_spent INTEGER,
            Transport_spent INTEGER,
            Entertainment_spent INTEGER,
            Others_spent INTEGER,     
            budget_expiry TEXT
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

def get_last_3_months_expenses(input_month):
    """
    Given an input month 'YYYY-MM', return a list of lists,
    each containing the total amount spent in each category for each of the previous 3 months (excluding the input month).
    The order of categories is: ['Education', 'Food', 'Healthcare', 'Housing', 'Others', 'Transport']
    If there is no data for a category, it is zero.
    If there is not enough data for 3 months:
      - If no months: return 3 lists of zeros.
      - If 1 month: repeat it 3 times.
      - If 2 months: fill the missing with the nearest available month.
    """
    year, month = map(int, input_month.split('-'))
    base_date = datetime.date(year, month, 1)
    months = []
    for _ in range(3):
        prev_month = base_date - datetime.timedelta(days=1)
        prev_month = prev_month.replace(day=1)
        months.append(prev_month.strftime('%Y-%m'))
        base_date = prev_month

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    month_data = []
    for m in months:
        category_totals = {cat: 0.0 for cat in CATEGORIES}
        cursor.execute("""
            SELECT category, SUM(amount) FROM expenses
            WHERE substr(date, 1, 7) = ?
            GROUP BY category
        """, (m,))
        rows = cursor.fetchall()
        for cat, amt in rows:
            if cat in category_totals and amt is not None:
                category_totals[cat] = amt
        month_data.append([category_totals[cat] for cat in CATEGORIES])
    conn.close()

    # Check for available data
    nonzero = [d for d in month_data if any(x != 0.0 for x in d)]
    if len(nonzero) == 0:
        return [[0.0]*len(CATEGORIES) for _ in range(3)]
    elif len(nonzero) == 1:
        return [nonzero[0]]*3
    elif len(nonzero) == 2:
        # Fill missing with nearest (repeat first or last)
        filled = []
        idx = 0
        for d in month_data:
            if any(x != 0.0 for x in d):
                filled.append(d)
                idx += 1
            else:
                # Use previous if possible, else next
                if idx == 0:
                    filled.append(nonzero[0])
                else:
                    filled.append(nonzero[1])
        return filled
    else:
        return month_data
import sqlite3

DB_NAME = "your_db_name.db"

def create_or_update_budget(
    budget_id=1,
    Food_budget=0,
    Education_budget=0,
    Healthcare_budget=0,
    Housing_budget=0,
    Transport_budget=0,
    Others_budget=0,
    budget_expiry=None
):
    """
    Create or update a budget entry in the budgets table.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Check if budget with this budget_id exists
    cursor.execute("SELECT budget_id FROM budgets WHERE budget_id = ?", (budget_id,))
    existing = cursor.fetchone()

    if existing:
        # Update existing budget
        cursor.execute("""
            UPDATE budgets
            SET Food_budget = ?, Education_budget = ?, Healthcare_budget = ?, 
                Housing_budget = ?, Transport_budget = ?, Others_budget = ?, 
                budget_expiry = ?
            WHERE budget_id = ?
        """, (Food_budget, Education_budget, Healthcare_budget, Housing_budget, 
              Transport_budget, Others_budget, budget_expiry, budget_id))
        print(f"Updated budget with id {budget_id}.")
    else:
        # Insert new budget
        cursor.execute("""
            INSERT INTO budgets 
            (budget_id, Food_budget, Education_budget, Healthcare_budget, 
             Housing_budget, Transport_budget, Others_budget, budget_expiry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (budget_id, Food_budget, Education_budget, Healthcare_budget, Housing_budget, 
              Transport_budget, Others_budget, budget_expiry))
        print(f"Created new budget with id {budget_id}.")

    conn.commit()
    conn.close()
# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    # init_db()

    # add_entry("2025-09-04", "spent 500 at kfc", "Food", 500)
    # add_entry("2025-09-04", "paid 2000 tuition fee", "Education", 2000)

    # print("Total on 2025-09-04:", get_total_by_date("2025-09-04"))
    # print("Category dict:", get_category_dict("2025-09-04"))
    # print("Raw entries:", get_entries("2025-09-04"))
    init_budget_table()

