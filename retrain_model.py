"""
Quick script to retrain the Random Forest model with current scikit-learn version
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("Starting model retraining...")
print("=" * 60)

# 1. Load cleaned data
print("\n1. Loading cleaned dataset...")
df = pd.read_csv('data/processed/cleaned_car_data.csv')
print(f"   ✓ Loaded {df.shape[0]} samples with {df.shape[1]} columns")

# 2. Prepare features and target
print("\n2. Preparing features and target...")
X = df.drop(columns=['price'], errors='ignore')
y = df['price']

# One-hot encode model column
X = pd.get_dummies(X, columns=['model'], drop_first=True)
print(f"   ✓ Feature shape after encoding: {X.shape}")

# 3. Train-test split
print("\n3. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   ✓ Training samples: {X_train.shape[0]}")
print(f"   ✓ Testing samples: {X_test.shape[0]}")

# 4. Train Random Forest model
print("\n4. Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)
print("   ✓ Training complete!")

# 5. Evaluate model
print("\n5. Evaluating model...")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"   ✓ R² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"   ✓ MAE: Rs. {mae:,.2f}")
print(f"   ✓ RMSE: Rs. {rmse:,.2f}")

# 6. Save model and metadata
print("\n6. Saving model artifacts...")

# Save model
model_path = 'models/random_forest_car_price_model.pkl'
joblib.dump(rf_model, model_path)
print(f"   ✓ Model saved to: {model_path}")

# Save feature names
feature_names = X_train.columns.tolist()
feature_path = 'models/feature_names.json'
with open(feature_path, 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"   ✓ Feature names saved to: {feature_path}")

# Save metadata
import sklearn
metadata = {
    'model_type': 'RandomForestRegressor',
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'random_state': 42,
    'training_samples': X_train.shape[0],
    'testing_samples': X_test.shape[0],
    'n_features': X_train.shape[1],
    'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'scikit_learn_version': sklearn.__version__
}

metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Metadata saved to: {metadata_path}")

# Save evaluation metrics
metrics = {
    'r2_score': float(r2),
    'mean_absolute_error': float(mae),
    'root_mean_squared_error': float(rmse),
    'r2_percentage': float(r2 * 100),
    'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_samples': len(y_test)
}

metrics_path = 'models/evaluation_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"   ✓ Evaluation metrics saved to: {metrics_path}")

print("\n" + "=" * 60)
print("✅ Model retraining completed successfully!")
print("=" * 60)
print(f"\nModel is now compatible with scikit-learn {sklearn.__version__}")
print("You can now run the Streamlit app without version errors.")
