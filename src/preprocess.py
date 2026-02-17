import pandas as pd
import numpy as np

def clean_and_preprocess():
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    files = [
        "data/raw/nissan_cars.csv", "data/raw/suzuki_cars.csv", "data/raw/toyota_cars.csv", 
        "data/raw/daihatsu_cars.csv", "data/raw/honda_cars.csv", "data/raw/mitsubishi_cars.csv"
    ]
    
    # Load and combine all CSVs
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Original Row Count: {len(df)}")

    # ---------------------------------------------------------
    # 2. CLEAN NUMERICAL COLUMNS (Mileage & Engine Capacity)
    # ---------------------------------------------------------
    # Remove ' km', ' cc', and commas
    df['mileage'] = df['mileage'].astype(str).str.replace(' km', '').str.replace(',', '')
    df['engine_capacity'] = df['engine_capacity'].astype(str).str.replace(' cc', '').str.replace(',', '')

    # Convert to Integer (coerce errors to NaN)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')

    # ---------------------------------------------------------
    # 3. STANDARDIZE MODEL NAMES (Fix Spelling/Consistency)
    # ---------------------------------------------------------
    # Convert to Title Case and strip whitespace
    df['model'] = df['model'].astype(str).str.strip().str.title()

    # Manual Fixes for known inconsistencies found in your data
    # Example: Merging "Roox Highway Star X" into "Roox" if desired, 
    # but keeping distinct variants like "Wagon R Stingray" is usually better for price prediction.
    # We will remove vague models like "Maruti" which is a brand name.
    
    # ---------------------------------------------------------
    # 4. FILTER DOMAIN (Inconsistencies Check)
    # ---------------------------------------------------------
    # Filter Brands
    valid_brands = ['Toyota', 'Suzuki', 'Honda', 'Nissan', 'Mitsubishi', 'Daihatsu']
    df = df[df['brand'].isin(valid_brands)]

    # Filter Price (Max 10 Million)
    df = df[df['price'] <= 10000000]

    # Filter Year (After 2005)
    df = df[df['manufacture_year'] > 2005]

    # Filter Body Type (Hatchback only)
    # Note: Some non-hatchbacks are mislabeled as Hatchback in raw data.
    # We also filter by a list of known non-hatchback models found in your files.
    non_hatchback_models = [
        'Vezel', 'Harrier', 'Crv', 'Xbee', 'Raize', 'Juke', 'Chr', 
        'Rush', 'S-Cross', 'Vitara', 'Sx4', 'Cross', 'Outlander', 
        'Pajero', 'S660', 'Copen'
    ]
    df = df[~df['model'].isin(non_hatchback_models)]
    df = df[df['body_type'] == 'Hatchback']

    # Filter Fuel Type (Petrol or Hybrid only)
    df = df[df['fuel_type'].isin(['Petrol', 'Hybrid'])]

    # Filter Transmission (Automatic only)
    # Even if data looks clean, this ensures consistency
    df = df[df['transmission'] == 'Automatic']

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['mileage', 'engine_capacity', 'price', 'model', 'fuel_type'])
    
    print(f"Row Count after Cleaning: {len(df)}")

    # ---------------------------------------------------------
    # 5. ONE-HOT ENCODING (Fuel Type)
    # ---------------------------------------------------------
    # IS IT GOOD PRACTICE? 
    # Yes. Since you only have 'Petrol' and 'Hybrid', this is standard.
    # We use drop_first=True to create a single column (0 or 1), 
    # which prevents multicollinearity (redundancy) and reduces overfitting.
    
    df = pd.get_dummies(df, columns=['fuel_type'], drop_first=True)
    # Result: 'fuel_type_Petrol' column (1=Petrol, 0=Hybrid)

    # ---------------------------------------------------------
    # 6. SAVE FINAL DATA
    # ---------------------------------------------------------
    # Select only the columns needed for your target
    # Note: 'brand' is not in your target list, but often useful to keep. 
    # We'll keep it for reference, but the model training might ignore it if you strictly want the list provided.
    final_columns = ['mileage', 'engine_capacity', 'manufacture_year', 'model', 'fuel_type_Petrol', 'price']
    
    # Note: 'model' is still text. Machine Learning models need numbers.
    # You will likely need to One-Hot Encode 'model' too during training, 
    # or use a Label Encoder, depending on the algorithm you choose.
    
    output_df = df[final_columns]
    output_df.to_csv("data/processed/cleaned_car_data.csv", index=False)
    print("Success! Processed data saved to 'data/processed/cleaned_car_data.csv'")

if __name__ == "__main__":
    clean_and_preprocess()