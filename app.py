from flask import Flask, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():

    features = [148]
    ip_features = np.array(features).reshape(1, -1)
    pred = model.predict(ip_features)

    return pred

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)