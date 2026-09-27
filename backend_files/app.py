
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model pipeline
# The path should be relative to where app.py will be executed or an absolute path
model_path = 'best_model/tuned_xgb_regressor_pipeline.joblib'
loaded_pipeline = joblib.load(model_path)

# Define the preprocessing logic that matches the training pipeline
def preprocess_data(data_df):
    # Ensure data_df is a copy to prevent SettingWithCopyWarning
    data_df = data_df.copy()

    # 1. Cleaning Product_Sugar_Content
    data_df.loc[:, 'Product_Sugar_Content'] = data_df['Product_Sugar_Content'].replace(to_replace=["reg"], value=["Regular"])

    # 2. Extracting Product_Id_char
    data_df.loc[:, 'Product_Id_char'] = data_df['Product_Id'].str[:2]

    # 3. Calculating Store_Age_Years
    data_df.loc[:, 'Store_Age_Years'] = 2025 - data_df['Store_Establishment_Year']

    # 4. Grouping Product Types into Perishables and Non-Perishables
    perishables = [
        "Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"
    ]
    data_df.loc[:, 'Product_Type_Category'] = data_df['Product_Type'].apply(lambda x: "Perishables" if x in perishables else "Non Perishables")
    
    # The loaded_pipeline (which is a ColumnTransformer + XGBRegressor) will handle
    # one-hot encoding of categorical features and feature selection internally.
    return data_df

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    if request.is_json:
        new_data_json = request.get_json()
        
        # Convert incoming JSON to DataFrame
        # Ensure consistent column order as in training data. 
        # This is crucial because OneHotEncoder in ColumnTransformer expects specific columns.
        # The `data` DataFrame from the notebook is our reference for column order (excluding target).
        # The raw input columns to the full pipeline are:
        # Product_Id, Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_Type,
        # Product_MRP, Store_Id, Store_Establishment_Year, Store_Size, Store_Location_City_Type, Store_Type

        # Ensure all expected raw columns are present, fill missing with None or NaN if necessary for robustness
        expected_raw_columns = [
            'Product_Id', 'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
            'Product_Type', 'Product_MRP', 'Store_Id', 'Store_Establishment_Year',
            'Store_Size', 'Store_Location_City_Type', 'Store_Type'
        ]
        
        # Convert list of dictionaries to DataFrame, ensuring expected columns
        new_data_df = pd.DataFrame(new_data_json, columns=expected_raw_columns)

        # Apply preprocessing
        processed_new_data = preprocess_data(new_data_df)

        # Make prediction using the loaded pipeline
        predictions = loaded_pipeline.predict(processed_new_data)

        return jsonify({'predicted_sales': predictions.tolist()})
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

@app.route('/predict_sales_batch', methods=['POST'])
def predict_sales_batch():
    if request.is_json:
        batch_data_json = request.get_json()
        
        # Convert list of dictionaries to DataFrame
        expected_raw_columns = [
            'Product_Id', 'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
            'Product_Type', 'Product_MRP', 'Store_Id', 'Store_Establishment_Year',
            'Store_Size', 'Store_Location_City_Type', 'Store_Type'
        ]
        batch_data_df = pd.DataFrame(batch_data_json, columns=expected_raw_columns)

        # Apply preprocessing
        processed_batch_data = preprocess_data(batch_data_df)

        # Make predictions
        predictions = loaded_pipeline.predict(processed_batch_data)

        return jsonify({'predicted_sales': predictions.tolist()})
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == '__main__':
    # For development, you might run like this:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI
    # For this Colab environment, we'll write the file and simulate deployment.
    pass # The app will be run via gunicorn in the Dockerfile
