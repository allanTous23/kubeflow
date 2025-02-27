from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
from preprocess import clean_data
from train_data import mainPlot,load_csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)



@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Vérifier si la requête contient un fichier
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier n'a été fourni"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "Aucun fichier n'a été sélectionné"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Seuls les fichiers CSV sont acceptés"}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "temp.csv")
        file.save(filepath)
        
        input_df = pd.read_csv(filepath)
        train_paths = [input_df]

        os.remove(filepath)
        
        train_df = clean_data(load_csv(train_paths))

        resultatsPca = mainPlot(train_df)

        dataPca = resultatsPca['pca_results']['X_pca']
        predictions = model.predict(dataPca)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "observation": i+1,
                "probabilities": pred.tolist() if hasattr(pred, "tolist") else float(pred),
                "predicted_class": int(np.argmax(pred)) if hasattr(pred, "__iter__") else None
            })
        
        return jsonify({"predictions": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)