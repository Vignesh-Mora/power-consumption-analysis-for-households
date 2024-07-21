from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('PCA_nodrl.pk1', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/input')
def input():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from form
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    # Define feature names
    features_name = [
        'Global_reactive_power', 'Voltage', 'Global_intensity', 
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'Year', 'Month', 'Sub_metering_4'
    ]

    # Create DataFrame
    df = pd.DataFrame(features_value, columns=features_name)

    # Make prediction
    output = model.predict(df)

    return render_template('result.html', prediction_text=f'Predicted Power Consumption: {output[0]}')

if __name__ == '__main__':
    app.run(debug=True)
