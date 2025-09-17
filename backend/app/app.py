import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import sqlite3
import os
import re
from ..add_to_db import add_entry, get_total_by_date,delete_entry_by_id
from torch.nn import functional as F

# Configure logger
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
#DB path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_NAME = os.path.join(DATA_DIR, "expenses.db")

# -------------------
# Define model
# -------------------
class ExpenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# -------------------
# Parameters
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/app
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")      # backend/models

vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
CATEGORY_MAPPING = torch.load(os.path.join(MODEL_DIR, "encoder.pth"))  # list of classes

INPUT_SIZE = len(vectorizer.get_feature_names_out())
HIDDEN_SIZE = 128
NUM_CLASSES = len(CATEGORY_MAPPING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ExpenseClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
state_dict = torch.load(os.path.join(MODEL_DIR, "category_predictor_model.pth"), map_location=device)
model.load_state_dict(state_dict)
model.eval()

# -------------------
# FastAPI app
# -------------------
app = FastAPI()

# -------------------
# Request schema
# -------------------
class ExpenseRequest(BaseModel):
    text: str
    date: str

def normalize_text(text: str) -> str:
    # Replace all numbers (like 5, 100, 2500) with <NUM>
    return re.sub(r"\d+", "<NUM>", text)
# -------------------
# Helpers
# -------------------
def extract_amount(text: str) -> float:
    matches = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
    return float(matches[0].replace(",", "")) if matches else 0.0

def predict_category_and_amount(text: str, threshold: float = 0.6):
    text_without_numbers = normalize_text(text)
    vec = vectorizer.transform([text_without_numbers]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32).to(device)

    output = model(vec)
    probs = F.softmax(output, dim=1)
    max_prob, pred_idx = torch.max(probs, dim=1)

    if max_prob.item() < threshold:
        pred_category = "Others"
    else:
        pred_category = CATEGORY_MAPPING[pred_idx.item()]
    logger.info(f"Predicted: {pred_category} with prob {max_prob.item():.4f}")
    return pred_category, extract_amount(text)

# -------------------
# FastAPI route
# -------------------
@app.post("/predict")
async def predict_expense(req: ExpenseRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="No text provided")
    if not req.date:
        raise HTTPException(status_code=400, detail="No date provided")

    category, amount = predict_category_and_amount(req.text)

    try:
        add_entry(req.date, req.text, category, amount)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB save failed: {str(e)}")

    total_amount = get_total_by_date(req.date)
    logger.info(f"Total for {req.date}: {total_amount}")
    return {"Today's Total": total_amount}
@app.delete("/expenses/{entry_id}")
def delete_expense(entry_id: int):
    # First, get the date of the expense before deleting
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT date FROM expenses WHERE id = ?", (entry_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"No expense found with ID {entry_id}")
    expense_date = row[0]
    # Delete the expense
    delete_entry_by_id(entry_id)

    # Get the updated total for that date
    updated_total = get_total_by_date(expense_date)

    return {
        "message": f"Expense with ID {entry_id} deleted successfully",
        "date": expense_date,
        "updated_total": updated_total
    }
@app.get("/expenses")
def get_all_expenses(date: str = None):
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if date:
        cursor.execute("SELECT id, text, category, amount FROM expenses WHERE TRIM(date) = ?", (date,))
    else:
        cursor.execute("SELECT id, text, category, amount FROM expenses")
    rows = cursor.fetchall()
    print("Querying for date:", date)
    print("Rows fetched from DB:", rows)
    
    conn.close()
 
    return [{"id": row[0], "text": row[1]} for row in rows]
