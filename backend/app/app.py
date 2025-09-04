import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import os
import re
from ..add_to_db import add_entry, get_total_by_date
from torch.nn import functional as F

# Configure logger
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


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
    return {"total_amount": total_amount}
