import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
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
from datetime import datetime
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
DB_NAME = os.path.join(DATA_DIR, "financedata.db")

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

        cleaned_items = clean_parsed_receipt(parsed_json)  # [{'name':..., 'quantity':..., 'price':...}, ...]

        # Predict category per item
        for item in cleaned_items:
            text = f"{item['quantity']} x {item['name']} @ {item['price']}"
            category, _ = predict_category_and_amount(text)
            item['category'] = category

        display_text, total = format_receipt_for_display(cleaned_items)

        return JSONResponse(content={
            "items": cleaned_items,
            "total": total
        })
    except Exception as e:
        logger.error("Error in /parse_receipt: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
def clean_parsed_receipt(parsed):
    cleaned = []
    for item in parsed:
        try:
            name = item.get("nm")
            if isinstance(name, dict):
                # Some garbage nested dict → skip
                continue
            if not name:
                continue

            quantity = item.get("cnt", 1)

            # Handle price (string like "85,00" → float 85.00)
            raw_price = str(item.get("price", "0")).replace(",", ".")
            try:
                price = float(raw_price)
            except ValueError:
                continue

            cleaned.append({
                "name": name.strip(),
                "quantity": int(quantity),
                "price": price
            })
        except Exception:
            continue
    format_receipt_for_display(cleaned)
    logger.info("Cleaned receipt data: %s", cleaned)
    return cleaned
def format_receipt_for_display(cleaned_items):
    lines = []
    total = 0.0
    for item in cleaned_items:
        line = f"{item['quantity']} x {item['name']} @ {item['price']:.2f}"
        lines.append(line)
        total += item['quantity'] * item['price']
    display_text = "\n".join(lines)
    logger.info("Formatted receipt data: %s", display_text)
    return display_text, total

@app.post("/receipt/confirm")
async def confirm_receipt(data: dict = Body(...)):
    items = data.get("items", [])
    if not items:
        raise HTTPException(status_code=400, detail="No items provided")

    date = datetime.datetime.today().strftime("%Y-%m-%d")
    saved_entries = []

    for it in items:
        name = it.get("name", "Unknown")
        quantity = int(it.get("quantity", 1))
        price = float(it.get("price", 0.0))
        category = it.get("category", "Others")

        total_price = quantity * price
        text = f"{quantity} x {name} @ {price:.2f}"

        add_entry(date, text, category, total_price)

        saved_entries.append({
            "name": name,
            "quantity": quantity,
            "price": price,
            "category": category,
            "total": total_price,
        })

    return {
        "message": "Receipt saved",
        "date": date,
        "entries": saved_entries,
    }

def extract_amount(text: str) -> float:
    amounts = [float(m.replace(',', '')) for m in re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)]
    return max(amounts) if amounts else 0.0
    

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
async def predict_category(req: ExpenseRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="No text provided")
    if not req.date:
        raise HTTPException(status_code=400, detail="No date provided")
    
    logger.info(f"Received text: {req.text} for date: {req.date}")
    category, amount = predict_category_and_amount(req.text)

    try:
        add_entry(req.date, req.text, category, amount)
        logger.info(f"Saved entry: {req.text}, {category}, {amount} on {req.date}")
    except Exception as e:
        logger.error(f"Error saving to DB: {e}")
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
    logger.info(f"Deleted expense with ID {entry_id} for date {expense_date}")
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
    logger.info("Querying for date: %s", date)
    logger.info("Rows fetched from DB: %s", rows)

    conn.close()
 
    return [{"id": row[0], "text": row[1]} for row in rows]

@app.get("/predict_expense")
async def predicted_expense(date: str): #YYYY-MM format
    if not date:
        raise HTTPException(status_code=400, detail="No date provided")
    
    last_3_month_expenses = get_last_3_months_expenses(date)
    logger.info("Fetched last 3 months expenses")
    if(sum(sum(row) for row in last_3_month_expenses)==0):
        logger.warning("No expense data for the last 3 months")
        raise HTTPException(status_code=404, detail="No expense data for the last 3 months")
    
    class ExpensePredictor(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(ExpensePredictor, self).__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    # Model parameters (should match those used during training)
    input_size = 6
    hidden_size = 16
    output_size = 6
    model_path = "../models/expense_predictor_model.pth"
    # Instantiate the model and load weights
    model = ExpensePredictor(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    last_3_months_tensor = torch.tensor(last_3_month_expenses).unsqueeze(0)  # shape (1, 3, 6)

    with torch.no_grad():
        prediction = model(last_3_months_tensor)
        logger.info("Predicted next month (normalized): %s", prediction.numpy())

@app.get("/create_update_budget")
def create_budget(budget: dict):
    try:
        create_or_update_budget(
            budget_id = budget.get("budget_id", 1),
            Food_budget=budget.get("Food_budget", 0),
            Education_budget=budget.get("Education_budget", 0),
            Healthcare_budget=budget.get("Healthcare_budget", 0),
            Housing_budget=budget.get("Housing_budget", 0),
            Transport_budget=budget.get("Transport_budget", 0),
            Others_budget=budget.get("Others_budget", 0),
            budget_expiry=budget.get("budget_expiry", None)
        )
        logger.info("Budget created/updated successfully")
        return {"message": "Budget created/updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
