import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Import Classification and Regression model files
with bz2.BZ2File('model/classification.pkl', 'rb') as pickle_in:
    model_C = pickle.load(pickle_in)

with bz2.BZ2File('model/regression.pkl', 'rb') as R_pickle_in:
    model_R = pickle.load(R_pickle_in)

# Load sample data (replace with actual training data range if available)
# Ensure the scaler is fitted on the actual dataset and not default values
df = pd.DataFrame({
    'Temperature': [15, 25, 35, 45],  # Example data points
    'Ws': [5, 10, 15, 20],
    'FFMC': [60, 75, 85, 90],
    'DMC': [30, 50, 70, 90],
    'ISI': [0.5, 2.0, 5.0, 10.0],
    'FWI': [5, 10, 15, 20],
    'Classes': [0, 1, 0, 1]  # Fire (1) or No Fire (0)
})

# Fit the scaler to actual data
X = df.drop(['FWI', 'Classes'], axis=1)
scaler = StandardScaler()
scaler.fit(X)  # Fit the scaler to the actual data for correct scaling

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for Classification Model Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = [float(x) for x in request.form.values()]
        final_features = [np.array(data)]
        
        # Log the input data for debugging
        print(f"Raw Input Data: {data}")
        
        # Transform data using the scaler
        final_features = scaler.transform(final_features)
        print(f"Scaled Input Data: {final_features}")  # Log scaled data
        
        # Predict using the classification model
        output = model_C.predict(final_features)[0]
        print(f"Classification Output: {output}")  # Log prediction

        # Output message based on the prediction
        if output == 0:
            text = 'Forest is Safe!'
            message = "{} --- No Fire Danger Detected".format(text)
        else:
            text = 'Forest is in Danger!'
            message = "{} --- There is a Chance of Fire".format(text)

        return render_template('index.html', prediction_text1=message)
    
    except Exception as e:
        return render_template('index.html', prediction_text1="Check the Input again!!!", error=str(e))

# Route for Regression Model Prediction
@app.route('/predictR', methods=['POST'])
def predictR():
    try:
        # Get input data from form
        data = [float(x) for x in request.form.values()]
        final_features = [np.array(data)]

        # Log the input data for debugging
        print(f"Raw Input Data: {data}")
        
        # Transform data using the scaler
        final_features = scaler.transform(final_features)
        print(f"Scaled Input Data: {final_features}")  # Log scaled data
        
        # Predict using the regression model
        output = model_R.predict(final_features)[0]
        print(f"Regression Output: {output}")  # Log prediction

        # Output message based on the prediction
        if output > 15:
            text = 'Forest is in Danger!'
            message = "{} --- There is a Chance of Fire".format(text)
        else:
            text = 'Forest is Safe!'
            message = "{} --- No Fire Danger Detected".format(text)

        return render_template('index.html', prediction_text2=message)
    
    except Exception as e:
        return render_template('index.html', prediction_text2="Check the Input again!!!", error=str(e))

# Run the app in Debug mode
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
