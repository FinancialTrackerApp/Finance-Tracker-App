import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import sqlite3
from transformers import DonutProcessor, VisionEncoderDecoderModel
import os
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import re
import io
from ..db_functions import *
from torch.nn import functional as F
# receipt parsing model
donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
donut_model.to(device)
# Configure logger
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
try:
    print(pytesseract.get_tesseract_version())
except Exception as e:
    print("Tesseract not found:", e)
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
def parse_text_from_image(img: Image.Image) -> str:
    """Run OCR on a PIL image and return text."""
    text = pytesseract.image_to_string(img)
    return text

def extract_receipt_data(text: str):
    """Extract date, amounts, and items from OCR text."""
    # Simple regex patterns for amounts and dates
    amounts = [float(m.replace(',', '')) for m in re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)]
    
    # Example: try to detect dates in format YYYY-MM-DD, DD/MM/YYYY, etc.
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})|(\d{2}/\d{2}/\d{4})', text)
    date = date_match.group(0) if date_match else None

    # For now, treat every line as an item (improve later)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    items = []
    for line in lines:
        # Try to pair each line with a number if present
        amount_match = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', line)
        item_amount = float(amount_match.group(0).replace(',', '')) if amount_match else 0.0
        items.append({"name": line, "amount": item_amount})

    total = sum(amounts)
    return {"date": date, "items": items, "total": total}


# -------------------
# FastAPI route
# -------------------

@app.post("/parse_receipt")
async def parse_receipt(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        # Donut task prompt
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = donut_processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        pixel_values = donut_processor(img, return_tensors="pt").pixel_values

        outputs = donut_model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=donut_model.decoder.config.max_position_embeddings,
            pad_token_id=donut_processor.tokenizer.pad_token_id,
            eos_token_id=donut_processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = donut_processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(donut_processor.tokenizer.eos_token, "").replace(
            donut_processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        parsed_json = donut_processor.token2json(sequence)
        return JSONResponse(content={"parsed": parsed_json})
    except Exception as e:
        logger.error("Error in /parse_receipt: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("predict/expense")
def predicted_expense(date: str): #YYYY-MM format
    if not date:
        raise HTTPException(status_code=400, detail="No date provided")
    last_3_month_expenses = get_last_3_months_expenses(date)
    if(sum(sum(row) for row in L)==0):
        raise HTTPException(status_code=404, detail="No expense data for the last 3 months")

@app.get("/stats/day/{date}")
def get_stats_for_day(date: str):
    try:
        return get_category_dict(date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/stats/daily")
def get_daily_totals():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT date, SUM(amount) FROM expenses GROUP BY date ORDER BY date")
    rows = cursor.fetchall()
    conn.close()
    return [{"date": row[0], "total": row[1]} for row in rows]