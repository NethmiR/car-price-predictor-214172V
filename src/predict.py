"""
Car Price Prediction Script
Uses the trained Random Forest model to predict car prices from input features.
"""

import joblib
import json
import pandas as pd
import numpy as np


def load_model_artifacts():
    """Load the trained model and related artifacts."""
    try:
        # Load the trained model
        model = joblib.load('models/random_forest_car_price_model.pkl')
        
        # Load feature names
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load metrics
        with open('models/evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print("✅ Model artifacts loaded successfully!")
        print(f"   Model trained on: {metadata['trained_date']}")
        print(f"   R² Score: {metrics['r2_score']:.4f}")
        print(f"   MAE: Rs. {metrics['mean_absolute_error']:,.2f}")
        
        return model, feature_names, metadata
    
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find model files. Please train the model first.")
        print(f"   Missing file: {e}")
        return None, None, None


def predict_price(model, feature_names, car_data):
    """
    Predict car price from input features.
    
    Parameters:
    -----------
    model : sklearn model
        Trained Random Forest model
    feature_names : list
        List of feature column names (must match training data)
    car_data : dict
        Dictionary containing car features:
        - mileage (int): Odometer reading in km
        - engine_capacity (int): Engine size in cc
        - manufacture_year (int): Year of manufacture
        - model (str): Car model name (e.g., 'Vitz', 'Swift')
        - fuel_type_Petrol (int): 1 if Petrol, 0 if Hybrid
    
    Returns:
    --------
    float : Predicted price in Rupees
    """
    
    # Create DataFrame with single row
    input_df = pd.DataFrame([car_data])
    
    # One-hot encode the model if it's a string
    if 'model' in input_df.columns:
        input_df = pd.get_dummies(input_df, columns=['model'], drop_first=True)
    
    # Ensure all features from training are present
    # Missing features will be set to 0 (model not selected)
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Ensure feature order matches training data
    input_df = input_df[feature_names]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return prediction


def main():
    """Example usage of the prediction system."""
    
    print("="*60)
    print("Car Price Prediction System")
    print("="*60)
    print()
    
    # Load model
    model, feature_names, metadata = load_model_artifacts()
    
    if model is None:
        return
    
    print("\n" + "-"*60)
    print("Example Predictions:")
    print("-"*60)
    
    # Example 1: Toyota Vitz 2018, Petrol
    car1 = {
        'mileage': 47000,
        'engine_capacity': 990,
        'manufacture_year': 2018,
        'model': 'Vitz',
        'fuel_type_Petrol': 1
    }
    
    price1 = predict_price(model, feature_names, car1)
    print("\nExample 1: Toyota Vitz")
    print(f"  Mileage: {car1['mileage']:,} km")
    print(f"  Engine: {car1['engine_capacity']} cc")
    print(f"  Year: {car1['manufacture_year']}")
    print(f"  Fuel: {'Petrol' if car1['fuel_type_Petrol'] else 'Hybrid'}")
    print(f"  → Predicted Price: Rs. {price1:,.0f}")
    
    # Example 2: Suzuki Swift 2017, Petrol
    car2 = {
        'mileage': 78000,
        'engine_capacity': 1000,
        'manufacture_year': 2017,
        'model': 'Swift',
        'fuel_type_Petrol': 1
    }
    
    price2 = predict_price(model, feature_names, car2)
    print("\nExample 2: Suzuki Swift")
    print(f"  Mileage: {car2['mileage']:,} km")
    print(f"  Engine: {car2['engine_capacity']} cc")
    print(f"  Year: {car2['manufacture_year']}")
    print(f"  Fuel: {'Petrol' if car2['fuel_type_Petrol'] else 'Hybrid'}")
    print(f"  → Predicted Price: Rs. {price2:,.0f}")
    
    # Example 3: Toyota Aqua 2014, Hybrid
    car3 = {
        'mileage': 120000,
        'engine_capacity': 1500,
        'manufacture_year': 2014,
        'model': 'Aqua',
        'fuel_type_Petrol': 0  # Hybrid
    }
    
    price3 = predict_price(model, feature_names, car3)
    print("\nExample 3: Toyota Aqua (Hybrid)")
    print(f"  Mileage: {car3['mileage']:,} km")
    print(f"  Engine: {car3['engine_capacity']} cc")
    print(f"  Year: {car3['manufacture_year']}")
    print(f"  Fuel: {'Petrol' if car3['fuel_type_Petrol'] else 'Hybrid'}")
    print(f"  → Predicted Price: Rs. {price3:,.0f}")
    
    print("\n" + "="*60)
    print("Prediction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
