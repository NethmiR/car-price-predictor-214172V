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
    # We use error_bad_lines=False (or on_bad_lines='skip' in newer pandas) to skip broken rows
    try:
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Original Row Count: {len(df)}")

    # ---------------------------------------------------------
    # 1.5. FIX SPELLING MISTAKES (New Step!)
    # ---------------------------------------------------------
    print("Fixing spelling mistakes...")
    
    # A dictionary to map incorrect names to correct ones
    # Format: 'Incorrect Name': 'Correct Name'
    corrections = {
        # Brands
        'Susuki': 'Suzuki',
        'Toyta': 'Toyota',
        'Toyato': 'Toyota',
        'Nisan': 'Nissan',
        'Hunda': 'Honda',
        
        # Models (Common mistakes based on your data)
        'Wagon R Fx': 'Wagon R FX',
        'Wagon R Fz': 'Wagon R FZ',
        'Wagonr': 'Wagon R',
        'Vits': 'Vitz',
        'Aqua G': 'Aqua',  # Simplify variants if needed
        'Prius C': 'Aqua', # Aqua is sold as Prius C in some regions
    }
    
    # Apply corrections to Brand and Model columns
    # We use .replace() with the dictionary
    df['brand'] = df['brand'].replace(corrections)
    df['model'] = df['model'].replace(corrections)

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
    # 3. STANDARDIZE MODEL NAMES
    # ---------------------------------------------------------
    # Convert to Title Case and strip whitespace
    df['model'] = df['model'].astype(str).str.strip().str.title()

    # ---------------------------------------------------------
    # 4. FILTER DOMAIN (Inconsistencies Check)
    # ---------------------------------------------------------
    # Filter Brands
    valid_brands = ['Toyota', 'Suzuki', 'Honda', 'Nissan', 'Mitsubishi', 'Daihatsu']
    df = df[df['brand'].isin(valid_brands)]

    # Filter Price (Max 10 Million)
    df = df[df['price'] <= 10000000]

    # Filter Year (After 2005)
    df = df[df['manufacture_year'] >= 2005]

    # Filter Body Type (Hatchback only)
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
    df = df[df['transmission'] == 'Automatic']

    # Drop rows with missing values
    df = df.dropna(subset=['mileage', 'engine_capacity', 'price', 'model', 'fuel_type'])
    
    print(f"Row Count after Cleaning: {len(df)}")

    # ---------------------------------------------------------
    # 5. ONE-HOT ENCODING (Fuel Type)
    # ---------------------------------------------------------
    df = pd.get_dummies(df, columns=['fuel_type'], drop_first=True)

    # ---------------------------------------------------------
    # 6. SAVE FINAL DATA
    # ---------------------------------------------------------
    final_columns = ['mileage', 'engine_capacity', 'manufacture_year', 'model', 'fuel_type_Petrol', 'price']
    output_df = df[final_columns]
    
    # Ensure directory exists before saving
    import os
    os.makedirs("data/processed", exist_ok=True)
    
    output_df.to_csv("data/processed/cleaned_car_data.csv", index=False)
    print("Success! Processed data saved.")

if __name__ == "__main__":
    clean_and_preprocess()