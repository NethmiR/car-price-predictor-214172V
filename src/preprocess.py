import pandas as pd
import numpy as np
import os

def clean_and_preprocess():
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    files = [
        "data/raw/nissan_cars.csv", "data/raw/suzuki_cars.csv", "data/raw/toyota_cars.csv", 
        "data/raw/daihatsu_cars.csv", "data/raw/honda_cars.csv", "data/raw/mitsubishi_cars.csv"
    ]
    
    # Load and combine all CSVs
    try:
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Original Row Count: {len(df)}")

    # ---------------------------------------------------------
    # 1.5. FIX SPELLING MISTAKES (Brand & Model)
    # ---------------------------------------------------------
    print("Fixing spelling mistakes...")
    corrections = {
        'Susuki': 'Suzuki', 'Toyta': 'Toyota', 'Toyato': 'Toyota', 
        'Nisan': 'Nissan', 'Hunda': 'Honda',
        'Wagon R Fx': 'Wagon R FX', 'Wagon R Fz': 'Wagon R FZ', 
        'Wagonr': 'Wagon R', 'Vits': 'Vitz', 'Aqua G': 'Aqua', 'Prius C': 'Aqua'
    }
    df['brand'] = df['brand'].replace(corrections)
    # Standardize model names first (Title Case)
    df['model'] = df['model'].astype(str).str.strip().str.title()
    df['model'] = df['model'].replace(corrections)

    # ---------------------------------------------------------
    # 1.6. AUTO-CORRECT BODY TYPE (New Step!)
    # ---------------------------------------------------------
    print("Auto-correcting body types for known hatchbacks...")
    
    # List of models that are DEFINITELY Hatchbacks
    # If a car is one of these, we force body_type = 'Hatchback'
    known_hatchbacks = [
        'Alto', 'Aqua', 'Baleno', 'Celerio', 'Dayz', 'Fit', 'Glanza', 'Hustler', 
        'Ignis', 'Ist', 'Leaf', 'March', 'Mira', 'Morning', 'Move', 'N-Box', 'N-Wgn', 
        'Note', 'Passo', 'Picanto', 'Pixis', 'Prius', 'Ractis', 'Spacia', 'Starlet', 
        'Swift', 'Tank', 'Tanto', 'Thor', 'Vitz', 'Wagon R', 'Wagon R FX', 
        'Wagon R FZ', 'Wagon R Stingray', 'Wigo', 'Yaris', 'EK Wagon', 'Cast Activa',
        'Roomy', 'Roox', 'Taft', 'A-Star'
    ]
    
    # Apply the fix:
    # If model is in known_hatchbacks, set body_type to 'Hatchback'
    df.loc[df['model'].isin(known_hatchbacks), 'body_type'] = 'Hatchback'

    # ---------------------------------------------------------
    # 2. CLEAN NUMERICAL COLUMNS
    # ---------------------------------------------------------
    df['mileage'] = df['mileage'].astype(str).str.replace(' km', '').str.replace(',', '')
    df['engine_capacity'] = df['engine_capacity'].astype(str).str.replace(' cc', '').str.replace(',', '')

    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')

    # ---------------------------------------------------------
    # 3. FILTER DOMAIN (Updated Rules)
    # ---------------------------------------------------------
    # Filter Brands
    valid_brands = ['Toyota', 'Suzuki', 'Honda', 'Nissan', 'Mitsubishi', 'Daihatsu']
    df = df[df['brand'].isin(valid_brands)]

    # Filter Price (<= 10 Million)
    df = df[df['price'] <= 10000000]

    # Filter Year (>= 2005)
    df = df[df['manufacture_year'] >= 2005]

    # Filter Body Type (Hatchback only)
    # Note: We do this AFTER the auto-correct step above.
    # We still check for non-hatchback models just in case.
    non_hatchback_models = [
        'Vezel', 'Harrier', 'Crv', 'Xbee', 'Raize', 'Juke', 'Chr', 
        'Rush', 'S-Cross', 'Vitara', 'Sx4', 'Cross', 'Outlander', 
        'Pajero', 'S660', 'Copen'
    ]
    df = df[~df['model'].isin(non_hatchback_models)]
    df = df[df['body_type'] == 'Hatchback']

    # Filter Fuel & Transmission
    df = df[df['fuel_type'].isin(['Petrol', 'Hybrid'])]
    df = df[df['transmission'] == 'Automatic']

    # Drop rows with missing values
    df = df.dropna(subset=['mileage', 'engine_capacity', 'price', 'model', 'fuel_type'])
    
    print(f"Row Count after Cleaning: {len(df)}")

    # ---------------------------------------------------------
    # 4. ONE-HOT ENCODING
    # ---------------------------------------------------------
    df = pd.get_dummies(df, columns=['fuel_type'], drop_first=True)

    # ---------------------------------------------------------
    # 5. SAVE FINAL DATA
    # ---------------------------------------------------------
    final_columns = ['mileage', 'engine_capacity', 'manufacture_year', 'model', 'fuel_type_Petrol', 'price']
    output_df = df[final_columns]
    
    os.makedirs("data/processed", exist_ok=True)
    output_df.to_csv("data/processed/cleaned_car_data.csv", index=False)
    print("Success! Processed data saved.")

if __name__ == "__main__":
    clean_and_preprocess()