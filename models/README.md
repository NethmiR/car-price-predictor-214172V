# Models Directory

This directory contains the trained machine learning model and related artifacts for the Car Price Prediction system.

## Files

### 1. `random_forest_car_price_model.pkl`

**Description**: The trained Random Forest Regressor model  
**Format**: Pickled scikit-learn model (joblib)  
**Size**: Varies based on model complexity  
**Usage**: Load with `joblib.load('models/random_forest_car_price_model.pkl')`

**Model Configuration**:

- Algorithm: Random Forest Regressor
- Number of Trees: 100
- Max Depth: 15
- Min Samples Split: 5
- Random State: 42

### 2. `feature_names.json`

**Description**: List of feature column names used during training  
**Format**: JSON array  
**Purpose**: Ensures correct feature order when making predictions

**Contents**:

- Base features: `mileage`, `engine_capacity`, `manufacture_year`, `fuel_type_Petrol`
- One-hot encoded model features: `model_Alto`, `model_Aqua`, `model_Baleno`, etc.

**Critical**: Predictions must use the exact same features in the same order.

### 3. `model_metadata.json`

**Description**: Model training configuration and statistics  
**Format**: JSON object

**Contents**:

```json
{
  "model_type": "RandomForestRegressor",
  "n_estimators": 100,
  "max_depth": 15,
  "min_samples_split": 5,
  "random_state": 42,
  "training_samples": <count>,
  "testing_samples": <count>,
  "n_features": <count>,
  "trained_date": "YYYY-MM-DD HH:MM:SS"
}
```

### 4. `evaluation_metrics.json`

**Description**: Model performance metrics on test set  
**Format**: JSON object

**Contents**:

```json
{
  "r2_score": <value>,
  "mean_absolute_error": <value>,
  "root_mean_squared_error": <value>,
  "r2_percentage": <value>,
  "evaluation_date": "YYYY-MM-DD HH:MM:SS",
  "test_samples": <count>
}
```

## Usage

### Loading the Model

```python
import joblib
import json

# Load model
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
```

### Making Predictions

**Option 1: Using the notebook**  
See section 6 in `notebooks/car_price_prediction_using_random_forest.ipynb`

**Option 2: Using the standalone script**

```bash
python src/predict.py
```

**Option 3: Custom implementation**

```python
import pandas as pd

# Prepare input data
car_data = {
    'mileage': 50000,
    'engine_capacity': 1000,
    'manufacture_year': 2018,
    'model': 'Vitz',
    'fuel_type_Petrol': 1
}

# Create DataFrame and encode
input_df = pd.DataFrame([car_data])
input_df = pd.get_dummies(input_df, columns=['model'], drop_first=True)

# Add missing features as zeros
for feature in feature_names:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Ensure correct order
input_df = input_df[feature_names]

# Predict
predicted_price = model.predict(input_df)[0]
print(f"Predicted Price: Rs. {predicted_price:,.0f}")
```

## Model Retraining

To retrain the model with updated data:

1. Update the dataset: `data/processed/cleaned_car_data.csv`
2. Run the notebook: `notebooks/car_price_prediction_using_random_forest.ipynb`
3. All model files will be automatically overwritten with new versions
4. Compare `evaluation_metrics.json` before and after to assess improvement

## Version Control

**Important**: These model files should be tracked in version control to:

- Maintain reproducibility
- Track model performance over time
- Enable rollback if new models perform poorly

**Note**: The `.pkl` file may be large. Consider using Git LFS for files > 100MB.

## Security & Validation

Before deploying the model to production:

1. ✅ Verify `evaluation_metrics.json` shows acceptable performance
2. ✅ Test predictions on known sample data
3. ✅ Check `model_metadata.json` for correct hyperparameters
4. ✅ Validate `feature_names.json` matches expected features
5. ✅ Ensure model version is documented

## Troubleshooting

**Error: "FileNotFoundError"**  
→ Run the training notebook to generate model files

**Error: "Feature mismatch"**  
→ Ensure input data has same features as `feature_names.json`

**Error: "Model gives unrealistic predictions"**  
→ Check input data is within training domain:

- Year ≥ 2005
- Price ≤ Rs. 10,000,000
- Model is a known hatchback
- Fuel type is Petrol or Hybrid

**Error: "Module not found: joblib"**  
→ Install dependencies: `pip install -r requirements.txt`

---

**Last Updated**: Auto-generated during model training  
**Contact**: Developer (Student ID: 214172V)
