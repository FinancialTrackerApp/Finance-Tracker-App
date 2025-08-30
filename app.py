from flask import Flask, request, jsonify
import import_ipynb
from model_code import cat_ip
print("flask app starting")
app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict_expense():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    updated_totals = cat_ip.predict(text)
    print(jsonify(updated_totals))
    return jsonify(updated_totals)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)