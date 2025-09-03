from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import os
import re

# -------------------
# Define model
# -------------------
class ExpenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ExpenseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# -------------------
# Parameters
# -------------------
INPUT_SIZE = 564  # must match training
HIDDEN_SIZE = 128
NUM_CLASSES = 6   # must match encoder/classes

# -------------------
# Flask app
# -------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/app
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")      # backend/models

# -------------------
# Load saved vectorizer
# -------------------
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# -------------------
# Load model
# -------------------
model_path = os.path.join(MODEL_DIR, "category_predictor_model.pth")
torch.serialization.add_safe_globals([ExpenseClassifier])
model = ExpenseClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -------------------
# Load encoder (category list)
# -------------------
encoder_path = os.path.join(MODEL_DIR, "encoder.pth")
CATEGORY_MAPPING = torch.load(encoder_path)  # exact class order from training
print("Category mapping:", CATEGORY_MAPPING)

# -------------------
# Running totals
# -------------------
totals = {cat: 0.0 for cat in CATEGORY_MAPPING}

# -------------------
# Prediction function
# -------------------
def predict_category_and_amount(text):
    # Vectorize input and move to device
    vec = vectorizer.transform([text]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32).to(device)

    # Predict
    output = model(vec)
    pred_idx = torch.argmax(output, 1).item()
    category = CATEGORY_MAPPING[pred_idx]

    # Extract amounts
    amount = extract_amount(text)
    totals[category] += amount

    print(f"Text: {text}")
    print(f"Predicted category: {category}")
    print(f"Amount extracted: {amount}")
    print(f"Updated totals: {totals}")

    return category, amount, totals

def extract_amount(text: str) -> float:
    # Look for any number in the text
    matches = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
    if matches:
        return float(matches[0].replace(",", ""))
    return 0.0

# -------------------
# Flask route
# -------------------
@app.route("/predict", methods=["POST"])
def predict_expense():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    _, _, updated_totals = predict_category_and_amount(text)
    return jsonify(updated_totals)

# -------------------
# Run server
# -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
