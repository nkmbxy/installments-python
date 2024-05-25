from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/predict-credit-scoring', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    input_data = pd.DataFrame([data])

    # Transform 'Marital Status' column only
    input_data['Marital Status'] = label_encoders['Marital Status'].transform(input_data['Marital Status'])
    input_data = scaler.transform(input_data)

    prediction = knn_model.predict(input_data)
    prediction_label = label_encoders['Credit Score'].inverse_transform(prediction)
    print(prediction_label)

    return jsonify({'creditScore': prediction_label[0]})

if __name__ == '__main__':
    app.run()
