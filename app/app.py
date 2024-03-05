from flask import Flask, request, jsonify, render_template
from joblib import load
import os
import sys
import pandas as pd
from sklearn import set_config

# Get the absolute path to the directory where app.py is located
app_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the joblib file
pipeline_path = os.path.join(app_dir, '../models/user_input_preprocessing_pipeline.joblib')
model_path = os.path.join(app_dir, '../models/vc_knn_gbc_sgd_standalone_balanced_trained.joblib')

processed_pipeline = load(pipeline_path)
model = load(model_path)

set_config(display='diagram')
set_config(transform_output='pandas')

app = Flask(__name__)

@app.route('/')
def make_health_check():
    return 'The server is up and running'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None

    if request.method == 'POST':
        data = request.form

        original_data = {
            'Gender': data['Gender'],
            'Vehicle_Damage': data['Vehicle_Damage'],
            'Vehicle_Age': data['Vehicle_Age'],
            'Region_Code': float(data['Region_Code']),
            'Policy_Sales_Channel': float(data['Policy_Sales_Channel']),
            'Annual_Premium': float(data['Annual_Premium']),
            'Age': int(data['Age']),
            'Previously_Insured': int(data['Previously_Insured'])
        }

        df = pd.DataFrame(original_data, index=[0])
        preprocessed_input = processed_pipeline.transform(df)

        prediction = model.predict(preprocessed_input)

        # Convert prediction to a more readable format
        prediction_text = 'Yes' if prediction[0] == 1 else 'No'

    return render_template('index.html', prediction={'Eligible Customer': prediction_text} if prediction_text else None)



@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()

    original_data = {
        'Gender': data['Gender'],
        'Vehicle_Damage': data['Vehicle_Damage'],
        'Vehicle_Age': data['Vehicle_Age'],
        'Region_Code': float(data['Region_Code']),
        'Policy_Sales_Channel': float(data['Policy_Sales_Channel']),
        'Annual_Premium': float(data['Annual_Premium']),
        'Age': int(data['Age']),
        'Previously_Insured': int(data['Previously_Insured'])
    }

    df = pd.DataFrame(original_data, index=[0])
    preprocessed_input = processed_pipeline.transform(df)

    prediction = model.predict(preprocessed_input)

    return jsonify({'prediction': int(prediction[0])})



if __name__ == '__main__':
    app.run(port=3000, debug=True)