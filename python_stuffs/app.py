# from flask import Flask, request, jsonify
# import import_ipynb
# from model_code import cat_ip
# print("flask app starting")
# app = Flask(__name__)
# @app.route("/predict", methods=["POST"])
# def predict_expense():
#     data = request.get_json()
#     text = data.get("text", "")
#     if not text:
#         return jsonify({"error": "No text provided"}), 400
#     updated_totals = cat_ip.predict(text)
#     print(jsonify(updated_totals))
#     return jsonify(updated_totals)
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.serialization
import spacy
import joblib
import os

# Define your model class here
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

# Model parameters
INPUT_SIZE = 563   # update to your actual input size
HIDDEN_SIZE = 64
NUM_CLASSES = 6

app = Flask(__name__)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load vectorizer
vectorizer_path = os.path.join("model_code", "pytorch_models", "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# Model path
model_path = os.path.join("model_code", "pytorch_models", "category_predictor_model.pth")

# Add safe globals for loading full model with custom class
torch.serialization.add_safe_globals([ExpenseClassifier])

# Instantiate model object with correct dimensions
model = ExpenseClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# Load weights into model instance
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

# Load label encoder (optional)
encoder_path = os.path.join("model_code", "pytorch_models", "encoder.pkl")
if os.path.exists(encoder_path):
    encoder = joblib.load(encoder_path)
else:
    encoder = None

# Running totals dictionary
totals = {
    "Transport": 0.0,
    "Healthcare": 0.0,
    "Food": 0.0,
    "Housing": 0.0,
    "Education": 0.0,
    "others": 0.0
}

def predict_category_and_amount(text):
    vec = vectorizer.transform([text]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32)

    output = model(vec)
    pred = torch.argmax(output, 1).item()

    if encoder:
        category = encoder.inverse_transform([pred])[0]
    else:
        category = str(pred)

    doc = nlp(text)
    amount = 0.0
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "CARDINAL"]:
            try:
                amount += float(ent.text)
            except:
                pass    

    if category in totals:
        totals[category] += amount
    else:
        totals["others"] += amount

    return category, amount, totals

@app.route("/predict", methods=["POST"])
def predict_expense():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    category, amount, updated_totals = predict_category_and_amount(text)

    return jsonify(
        updated_totals
    )

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)




