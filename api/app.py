from flask import Flask, request, jsonify
import requests
import mysql.connector
import os

app = Flask(__name__)

# Configuration de la DB
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "database": os.getenv("DB_NAME", "predictions"),
}

# URL du service modèle
MODEL_URL = os.getenv("MODEL_URL", "http://model:5001/predict")

@app.route('/', methods=["GET"])
def hello():
    return {"welcome": "hello world every one"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    response = requests.post(MODEL_URL, json=data)
    prediction = response.json()

    # Sauvegarde dans MySQL
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO predictions (input_data, output_data) VALUES (%s, %s);", 
                    (str(data), str(prediction)))
        conn.commit()
        cur.close()
        conn.close()
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
