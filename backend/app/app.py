from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import os
import re
from ..add_to_db import add_entry, get_total_by_date

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

# -------------------
# Helpers
# -------------------
def extract_amount(text: str) -> float:
    matches = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
    return float(matches[0].replace(",", "")) if matches else 0.0

def predict_category_and_amount(text: str, threshold: float = 0.7):
    vec = vectorizer.transform([text]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32).to(device)

    output = model(vec)
    probs = torch.softmax(output, dim=1)
    max_prob, pred_idx = torch.max(probs, dim=1)

    if max_prob.item() < threshold:
        pred_category = "Others"
    else:
        pred_category = CATEGORY_MAPPING[pred_idx.item()]

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
    return {"total_amount": total_amount}
