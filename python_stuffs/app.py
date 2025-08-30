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
import spacy
import pickle

# Load model, vectorizer, encoder
model = torch.load("model.pth", map_location="cpu")
model.eval()
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Totals dictionary
totals = {
    "transport": 0.0,
    "Healthcare": 0.0,
    "Food & Groceries": 0.0,
    "Housing & Utilities": 0.0,
    "Education": 0.0,
    "others": 0.0
}

app = Flask(__name__)

def predict_category_and_amount(text):
    # Vectorize input
    vec = vectorizer.transform([text]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32)

    # Model prediction
    output = model(vec)
    pred = torch.argmax(output, 1).item()
    category = encoder.inverse_transform([pred])[0]

    # Amount extraction with spaCy
    doc = nlp(text)
    amount = 0.0
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "CARDINAL"]:
            try:
                amount += float(ent.text)
            except:
                pass

    # Update totals
    totals[category] += amount
    return category, amount, totals

@app.route("/predict", methods=["POST"])
def predict_expense():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"})

    category, amount, updated_totals = predict_category_and_amount(text)

    return jsonify({
        "text": text,
        "category": category,
        "amount": amount,
        "totals": updated_totals
    })

if __name__ == "__main__":
    app.run(debug=True)


