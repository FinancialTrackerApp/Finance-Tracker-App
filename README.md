📊 FinanceKoi - Expense Tracker

A full-stack Expense Tracker app with:

Flutter frontend for input, visualization, and receipt upload

FastAPI backend with ML-powered text classification

SQLite database for local expense storage

PyTorch model for automatic category prediction

Donut model for receipt understanding (structured parsing, not just OCR)

(Optional) Neon Postgres integration — scaffolded but not yet active

🚀 Features

Enter expenses in natural language (e.g., "Spent 500 at KFC").

Upload receipts → Donut model extracts text + structured fields → auto-adds expenses.

ML model detects amounts and predicts categories (Food, Transport, Education, Housing, Healthcare, Others).

View daily totals and category breakdowns.

Delete entries and update totals instantly.

Frontend visualization with charts and calendar.

🧾 Receipt Handling (Donut Model)

Instead of raw OCR, receipts are parsed using Donut (Document Understanding Transformer).

Donut directly outputs structured JSON (e.g., { "items": [...], "total": 600, "merchant": "KFC" }).

Extracted data is mapped into date, category, amount, and text before storing in DB.

Example Flow

Upload receipt (e.g., "KFC_Bill.jpg").

Donut model parses →

{
  "merchant": "KFC",
  "items": [
    {"name": "Burger", "price": 500},
    {"name": "Coke", "price": 100}
  ],
  "total": 600,
  "date": "2025-09-04"
}


App saves expense: Food - 600 Rs (Spent at KFC).

Entry appears in daily summary & category breakdown.

🏗 Project Structure
backend/
│

├── app.py                  # FastAPI app (routes: /predict, /expenses, /stats, /receipt)

├── add_to_db.py            # SQLite helpers

├── database.py             # Neon DB scaffold (not yet active)

├── cat_ip.py               # ML training script (TF-IDF + PyTorch classifier)

├── data/

│   ├── expenses.db         # SQLite database

│   └── text_category.csv   # Training dataset

├── models/

│   ├── vectorizer.pkl

│   ├── category_predictor_model.pth

│   ├── encoder.pth

│   └── donut_model/        # Pretrained Donut weights

│

frontend/

└── main.dart               # Flutter app (UI, charts, calendar, receipt upload)

⚙️ Backend Setup
1. Install dependencies
pip install fastapi uvicorn torch scikit-learn joblib sqlite3 spacy
pip install transformers timm  # Donut dependencies
python -m spacy download en_core_web_sm

2. Initialize DB
python backend/add_to_db.py

3. Run backend
uvicorn backend.app:app --reload

🔑 API Routes

POST /predict
Add text-based expense → classify + save.

POST /receipt
Upload receipt image → Donut parses → auto-save expense.

GET /expenses
List all expenses (filter by date if needed).

DELETE /expenses/{id}
Delete expense by ID.

GET /stats/day/{date}
Category breakdown for a date.

✅ TODO

 Integrate Neon Postgres (currently scaffolded, not active).

 Improve Donut fine-tuning for regional receipt formats.

 Deploy backend + connect Flutter app.

