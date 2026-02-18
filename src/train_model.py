import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('data/processed/cleaned_car_data.csv')

# 2. Preprocessing for Model
# Ensure 'model' is treated as a string (Categorical)
df['model'] = df['model'].astype(str)

# Define Features (X) and Target (y)
X = df[['mileage', 'engine_capacity', 'manufacture_year', 'model', 'fuel_type_Petrol']]
y = df['price']

# Identify categorical features indices for CatBoost
categorical_features_indices = np.where(X.dtypes != float)[0]
# In your case, 'model' and 'fuel_type_Petrol' might be non-float. 
# We explicitly list 'model' index if we know it, or let the code find object/bool columns.
cat_features = ['model'] 

# 3. Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train CatBoost Regressor
# We use 'RMSE' as the loss function since we are predicting price
model = CatBoostRegressor(
    iterations=1000, 
    learning_rate=0.1, 
    depth=6, 
    loss_function='RMSE',
    verbose=100,
    cat_features=cat_features # Passing the categorical column directly
)

print("Training Model...")
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# 5. Evaluation (20 Marks)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Metrics:")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

# 6. Explainability with SHAP (20 Marks)
print("\nGenerating SHAP Explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot 1: Summary Plot (Global Importance)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Feature Importance (SHAP)")
plt.tight_layout()
plt.show()

# Plot 2: Dependence Plot (Effect of specific feature, e.g., Manufacture Year)
# This shows how year affects price, interacting with other features
shap.dependence_plot("manufacture_year", shap_values, X_test)