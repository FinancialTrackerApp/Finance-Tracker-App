    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    DB_NAME = os.path.join(DATA_DIR, "expenses.db")