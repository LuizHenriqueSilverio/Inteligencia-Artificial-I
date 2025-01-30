import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Captura os valores do formulário e converte para float
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Faz a predição
        pred = model.predict(final_features)
        output = pred[0]  # Obtemos o rótulo diretamente
        
        return render_template("index.html", prediction_text=f"Nível de obesidade previsto: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Erro na predição: {str(e)}")

@app.route("/api", methods=["POST"])
def results():
    try:
        data = request.get_json(force=True)
        pred = model.predict([np.array(list(data.values()))])
        output = pred[0]
        return jsonify({"obesity_level": output})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)