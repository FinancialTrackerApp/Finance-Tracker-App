from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.serialization
import spacy
import joblib
import os
import re

# Define model
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

# Model params
INPUT_SIZE = 560
HIDDEN_SIZE = 64

# Flask app
app = Flask(__name__)

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Load vectorizer
vectorizer_path = os.path.join("model_code", "pytorch_models", "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# Load model
model_path = os.path.join("model_code", "pytorch_models", "category_predictor_model.pth")
torch.serialization.add_safe_globals([ExpenseClassifier])
num_classes = 6
model = ExpenseClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Load encoder
encoder_path = os.path.join("model_code", "pytorch_models", "encoder.pth")
CATEGORY_MAPPING = torch.load(encoder_path)  # list of categories in correct order

# Running totals
totals = {cat: 0.0 for cat in CATEGORY_MAPPING}

# Prediction function
def predict_category_and_amount(text):
    vec = vectorizer.transform([text]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32)
    output = model(vec)
    pred_idx = torch.argmax(output, 1).item()
    category = CATEGORY_MAPPING[pred_idx]

    # Extract amounts
    amount = 0.0
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "CARDINAL"]:
            nums = re.findall(r'\d+', ent.text.replace(',', ''))
            if nums:
                amount += float(nums[0])

    totals[category] += amount

    print(f"Text: {text}")
    print(f"Predicted category: {category}")
    print(f"Amount extracted: {amount}")
    print(f"Updated totals: {totals}")

    return category, amount, totals

# Flask route
@app.route("/predict", methods=["POST"])
def predict_expense():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    _, _, updated_totals = predict_category_and_amount(text)
    return jsonify(updated_totals)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
