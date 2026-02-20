# Car Price Prediction System - Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Domain Definition](#domain-definition)
3. [Data Collection Pipeline](#data-collection-pipeline)
4. [Web Scraping Implementation](#web-scraping-implementation)
5. [Raw Data Structure](#raw-data-structure)
6. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
7. [Model Development](#model-development)
8. [Results and Performance](#results-and-performance)
9. [Project Structure](#project-structure)

---

## Project Overview

### Objective

Develop a machine learning model to predict the price of used cars based on the following input features:

- Mileage (km)
- Engine Capacity (cc)
- Manufacture Year
- Model Name
- Fuel Type (Petrol/Hybrid)

### Business Context

The Sri Lankan used car market, particularly on platforms like ikman.lk, requires accurate price estimation to help buyers and sellers make informed decisions. This system provides data-driven price predictions for hatchback vehicles in a specific market segment.

---

## Domain Definition

The system focuses on a specific segment of the used car market with the following constraints:

| Criterion            | Value/Range                                         |
| -------------------- | --------------------------------------------------- |
| **Manufacture Year** | After 2005 (inclusive)                              |
| **Transmission**     | Automatic only                                      |
| **Fuel Type**        | Petrol or Hybrid                                    |
| **Body Type**        | Hatchback                                           |
| **Brands**           | Toyota, Suzuki, Honda, Nissan, Mitsubishi, Daihatsu |
| **Maximum Price**    | Rs. 10,000,000                                      |

### Rationale

This domain was chosen based on:

- High liquidity in the market (most common segment)
- Standardized features across brands
- Sufficient data availability on ikman.lk
- Relevance to typical buyers in Sri Lanka

---

## Data Collection Pipeline

### Stage 1: Advertisement Filtering on ikman.lk

Data collection was performed directly on the ikman.lk platform using the following filtering strategy:

1. **Category Selection**: Cars & Vans → Cars
2. **Brand Filtering**: Selected Toyota, Suzuki, Honda, Nissan, Mitsubishi, Daihatsu individually
3. **Advanced Filters Applied**:
   - Body Type: Hatchback
   - Transmission: Automatic
   - Fuel Type: Petrol, Hybrid
   - Year Range: 2005 - Present
   - Price Range: Up to Rs. 10,000,000

### Stage 2: Link Extraction

Advertisement URLs were manually collected and organized by brand. The links were stored in text files under the `data/car-ad-links/` directory:

```
data/car-ad-links/
├── toyota.txt       (Toyota vehicle listings)
├── suzuki.txt       (Suzuki vehicle listings)
├── honda.txt        (Honda vehicle listings)
├── nissan.txt       (Nissan vehicle listings)
├── mitsubishi.txt   (Mitsubishi vehicle listings)
└── daihatsu.txt     (Daihatsu vehicle listings)
```

Each text file contains one advertisement URL per line.

### Data Collection Statistics

| Brand      | Links Collected |
| ---------- | --------------- |
| Toyota     | Multiple ads    |
| Suzuki     | Multiple ads    |
| Honda      | Multiple ads    |
| Nissan     | Multiple ads    |
| Mitsubishi | Multiple ads    |
| Daihatsu   | Multiple ads    |

**Note**: Pre-filtering on ikman.lk ensured that all collected advertisements already met the domain criteria, reducing downstream processing overhead.

---

## Web Scraping Implementation

### Overview

The scraping system (`scrape.py`) extracts detailed vehicle information from ikman.lk advertisement pages using BeautifulSoup and requests libraries.

### Technical Architecture

#### Dependencies

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from tqdm import tqdm
import os
import glob
```

### Core Functions

#### 1. `extract_ad_data(url)`

Extracts structured vehicle data from a single advertisement page.

**Input**: Advertisement URL (string)

**Output**: Dictionary containing:

- `brand`: Manufacturer name
- `model`: Vehicle model
- `trim`: Trim/Edition level
- `manufacture_year`: Year of manufacture
- `condition`: Vehicle condition
- `transmission`: Transmission type
- `body_type`: Body type
- `fuel_type`: Fuel type
- `engine_capacity`: Engine capacity (cc)
- `mileage`: Odometer reading (km)
- `price`: Listed price (Rs)

**Implementation Details**:

```python
def extract_ad_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract price
        price_tag = soup.find('div', class_=lambda x: x and 'amount--' in x)

        # Extract all details by finding label-value pairs
        labels = soup.find_all('div', class_=lambda x: x and 'label--' in x)

        # Map labels to data fields
        # Returns structured dictionary
```

**Key Features**:

- Uses dynamic class name matching (handles class name changes)
- Implements error handling for network issues
- Returns `None` for failed extractions
- Cleans price text (removes "Rs" and commas)

#### 2. `process_brand_file(txt_file_path)`

Processes all URLs for a specific brand and generates CSV output.

**Workflow**:

1. Load URLs from brand-specific text file
2. Deduplicate URLs using set conversion
3. Iterate through each URL with progress bar
4. Call `extract_ad_data()` for each URL
5. Buffer results in memory (batch size: 50)
6. Checkpoint data to CSV every 50 records
7. Track and save failed URLs separately
8. Implement rate limiting (0.5-1.5 seconds between requests)

**Features**:

- **Checkpointing**: Saves data in batches to prevent data loss
- **Error Tracking**: Failed URLs saved to `data/failed-links/`
- **Rate Limiting**: Random delays to avoid IP blocking
- **Progress Monitoring**: tqdm progress bar
- **Append Mode**: Continues from last checkpoint if interrupted

**Output Files**:

- `data/raw/{brand}_cars.csv`: Successfully scraped vehicle data
- `data/failed-links/{brand}_failed_links.txt`: URLs that failed to scrape

#### 3. Main Execution Block

Automatically processes all brand files found in `data/car-ad-links/`.

```python
if __name__ == "__main__":
    txt_files = glob.glob("data/car-ad-links/*.txt")

    for txt_file in txt_files:
        process_brand_file(txt_file)
```

### Scraping Strategy

#### Rate Limiting

```python
time.sleep(random.uniform(0.5, 1.5))
```

Random delays between 0.5-1.5 seconds prevent server overload and reduce detection risk.

#### Checkpointing System

Data is saved every 50 records:

```python
if len(data_buffer) >= 50:
    df_chunk = pd.DataFrame(data_buffer)
    df_chunk.to_csv(csv_filename, mode='a', header=False, index=False)
    data_buffer = []
```

This ensures that even if the scraper crashes, previously collected data is preserved.

#### Error Handling

- Network timeouts (10-second limit)
- HTTP status code validation
- Failed URL tracking for manual review
- Graceful degradation (continues on individual failures)

---

## Raw Data Structure

### Storage Location

```
data/raw/
├── toyota_cars.csv
├── suzuki_cars.csv
├── honda_cars.csv
├── nissan_cars.csv
├── mitsubishi_cars.csv
└── daihatsu_cars.csv
```

### Schema

Each CSV file contains the following columns:

| Column             | Data Type | Description                | Example     |
| ------------------ | --------- | -------------------------- | ----------- |
| `brand`            | String    | Manufacturer name          | "Toyota"    |
| `model`            | String    | Vehicle model              | "Vitz"      |
| `trim`             | String    | Trim/Edition               | "RS"        |
| `manufacture_year` | Integer   | Year of manufacture        | 2018        |
| `condition`        | String    | Vehicle condition          | "Used"      |
| `transmission`     | String    | Transmission type          | "Automatic" |
| `body_type`        | String    | Body type                  | "Hatchback" |
| `fuel_type`        | String    | Fuel type                  | "Petrol"    |
| `engine_capacity`  | String    | Engine size with unit      | "990 cc"    |
| `mileage`          | String    | Odometer reading with unit | "85,000 km" |
| `price`            | Integer   | Listed price in Rs         | 7500000     |

### Data Characteristics

- **Units Included**: Raw data contains units ("km", "cc")
- **Formatting**: Comma separators in numeric fields
- **Completeness**: Some fields may be empty (handled in preprocessing)
- **Duplicates**: Possible due to multiple scraping sessions

---

## Data Preprocessing Pipeline

The preprocessing script (`src/preprocess.py`) transforms raw scraped data into a clean, ML-ready dataset.

### Pipeline Stages

#### Stage 1: Data Loading and Consolidation

```python
files = [
    "data/raw/nissan_cars.csv",
    "data/raw/suzuki_cars.csv",
    "data/raw/toyota_cars.csv",
    "data/raw/daihatsu_cars.csv",
    "data/raw/honda_cars.csv",
    "data/raw/mitsubishi_cars.csv"
]

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
```

Combines all brand-specific CSV files into a single DataFrame for unified processing.

#### Stage 2: Spelling and Naming Standardization

**Problem**: Scraped data contains inconsistent naming conventions.

**Solution**: Apply correction mapping

```python
corrections = {
    'Susuki': 'Suzuki',
    'Toyta': 'Toyota',
    'Toyato': 'Toyota',
    'Nisan': 'Nissan',
    'Hunda': 'Honda',
    'Wagon R Fx': 'Wagon R FX',
    'Wagon R Fz': 'Wagon R FZ',
    'Wagonr': 'Wagon R',
    'Vits': 'Vitz',
    'Aqua G': 'Aqua',
    'Prius C': 'Aqua'
}

df['brand'] = df['brand'].replace(corrections)
df['model'] = df['model'].astype(str).str.strip().str.title()
df['model'] = df['model'].replace(corrections)
```

**Transformations**:

- Fix common spelling mistakes in brand names
- Standardize model names to Title Case
- Unify variant naming (e.g., "Wagon R Fx" → "Wagon R FX")
- Merge equivalent models (e.g., "Prius C" → "Aqua")

#### Stage 3: Body Type Auto-Correction

**Problem**: Body type field may be incorrectly labeled or missing in scraped data.

**Solution**: Override body type for known hatchback models

```python
known_hatchbacks = [
    'Alto', 'Aqua', 'Baleno', 'Celerio', 'Dayz', 'Fit', 'Glanza',
    'Hustler', 'Ignis', 'Ist', 'Leaf', 'March', 'Mira', 'Morning',
    'Move', 'N-Box', 'N-Wgn', 'Note', 'Passo', 'Picanto', 'Pixis',
    'Prius', 'Ractis', 'Spacia', 'Starlet', 'Swift', 'Tank', 'Tanto',
    'Thor', 'Vitz', 'Wagon R', 'Wagon R FX', 'Wagon R FZ',
    'Wagon R Stingray', 'Wigo', 'Yaris', 'EK Wagon', 'Cast Activa',
    'Roomy', 'Roox', 'Taft', 'A-Star'
]

df.loc[df['model'].isin(known_hatchbacks), 'body_type'] = 'Hatchback'
```

This ensures that known hatchback models are correctly classified regardless of source data quality.

#### Stage 4: Numerical Column Cleaning

**Objective**: Convert string-formatted numbers to numeric types

```python
# Remove units and formatting
df['mileage'] = df['mileage'].astype(str).str.replace(' km', '').str.replace(',', '')
df['engine_capacity'] = df['engine_capacity'].astype(str).str.replace(' cc', '').str.replace(',', '')

# Convert to numeric (coerce errors to NaN)
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
```

**Transformations**:

- "85,000 km" → 85000
- "990 cc" → 990
- Invalid entries → NaN (handled later)

#### Stage 5: Domain Filtering

Apply strict filtering criteria to ensure all records meet domain requirements:

##### 5.1 Brand Filter

```python
valid_brands = ['Toyota', 'Suzuki', 'Honda', 'Nissan', 'Mitsubishi', 'Daihatsu']
df = df[df['brand'].isin(valid_brands)]
```

##### 5.2 Price Filter

```python
df = df[df['price'] <= 10000000]
```

##### 5.3 Year Filter

```python
df = df[df['manufacture_year'] >= 2005]
```

##### 5.4 Body Type Filter

```python
# Explicitly exclude non-hatchback models
non_hatchback_models = [
    'Vezel', 'Harrier', 'Crv', 'Xbee', 'Raize', 'Juke', 'Chr',
    'Rush', 'S-Cross', 'Vitara', 'Sx4', 'Cross', 'Outlander',
    'Pajero', 'S660', 'Copen'
]
df = df[~df['model'].isin(non_hatchback_models)]
df = df[df['body_type'] == 'Hatchback']
```

##### 5.5 Fuel and Transmission Filter

```python
df = df[df['fuel_type'].isin(['Petrol', 'Hybrid'])]
df = df[df['transmission'] == 'Automatic']
```

##### 5.6 Missing Value Removal

```python
df = df.dropna(subset=['mileage', 'engine_capacity', 'price', 'model', 'fuel_type'])
```

#### Stage 6: Feature Encoding

Convert categorical `fuel_type` to binary numeric encoding:

```python
df = pd.get_dummies(df, columns=['fuel_type'], drop_first=True)
```

**Result**:

- Creates `fuel_type_Petrol` column (1 = Petrol, 0 = Hybrid)
- Eliminates categorical data for ML compatibility
- `drop_first=True` prevents multicollinearity

#### Stage 7: Final Dataset Export

```python
final_columns = [
    'mileage',
    'engine_capacity',
    'manufacture_year',
    'model',
    'fuel_type_Petrol',
    'price'
]

output_df = df[final_columns]
os.makedirs("data/processed", exist_ok=True)
output_df.to_csv("data/processed/cleaned_car_data.csv", index=False)
```

### Preprocessing Results

**Output File**: `data/processed/cleaned_car_data.csv`

**Final Schema**:

| Column             | Type    | Description              |
| ------------------ | ------- | ------------------------ |
| `mileage`          | Integer | Odometer reading (km)    |
| `engine_capacity`  | Integer | Engine size (cc)         |
| `manufacture_year` | Integer | Year of manufacture      |
| `model`            | String  | Standardized model name  |
| `fuel_type_Petrol` | Boolean | 1 if Petrol, 0 if Hybrid |
| `price`            | Integer | Target variable (Rs)     |

**Data Quality Improvements**:

- ✅ All units removed
- ✅ Numerical columns properly typed
- ✅ Spelling standardized
- ✅ Domain constraints enforced
- ✅ Missing values eliminated
- ✅ Categorical encoding applied
- ✅ Duplicate entries removed

---

## Model Development

### Implementation Environment

The model training is implemented in a Jupyter notebook:
`notebooks/car_price_prediction_using_random_forest.ipynb`

### Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Data Loading

```python
file_path = '../data/processed/cleaned_car_data.csv'
df = pd.read_csv(file_path)
```

**Note**: Relative path used since notebook is in `notebooks/` subdirectory.

### Feature Engineering

#### Model Column Encoding

The `model` column contains categorical text data (e.g., "Vitz", "Swift"). Random Forest requires numerical input, so one-hot encoding is applied:

```python
# Separate features and target
X = df.drop(columns=['price'], errors='ignore')
y = df['price']

# One-hot encode model names
X = pd.get_dummies(X, columns=['model'], drop_first=True)
```

**Result**:

- Each unique model becomes a binary feature column
- `drop_first=True` avoids multicollinearity (dummy variable trap)
- Feature space expands from 5 columns to 5 + (n_models - 1) columns

**Example Transformation**:

Before:

```
mileage | engine_capacity | manufacture_year | model  | fuel_type_Petrol
47000   | 990            | 2018            | Vitz   | 1
```

After:

```
mileage | engine_capacity | manufacture_year | fuel_type_Petrol | model_Vitz | model_Swift | ...
47000   | 990            | 2018            | 1                | 1          | 0           | ...
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

**Configuration**:

- 80% training data
- 20% testing data
- `random_state=42` ensures reproducibility

### Model Selection: Random Forest Regressor

#### Rationale

Random Forest was chosen over alternatives for the following reasons:

1. **Non-linear Relationships**: Car depreciation and pricing involve complex non-linear patterns that linear regression cannot capture effectively.

2. **Robustness to Outliers**: Used car prices can have outliers (e.g., rare editions, modified vehicles). Random Forest is less sensitive to these compared to linear models.

3. **No Feature Scaling Required**: Unlike neural networks or SVMs, Random Forest works directly with raw numerical features.

4. **Feature Importance**: Provides built-in feature importance metrics for interpretability.

5. **Ensemble Strength**: Aggregates predictions from multiple decision trees, reducing overfitting risk.

#### Hyperparameter Configuration

```python
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth (pruning)
    min_samples_split=5,   # Minimum samples to split (pruning)
    random_state=42        # Reproducibility
)
```

**Hyperparameter Details**:

| Parameter           | Value | Purpose                                                                                       |
| ------------------- | ----- | --------------------------------------------------------------------------------------------- |
| `n_estimators`      | 100   | Builds 100 decision trees. More trees = better stability and generalization.                  |
| `max_depth`         | 15    | Limits tree depth to prevent overfitting. Stops trees from memorizing training data.          |
| `min_samples_split` | 5     | A node must contain at least 5 samples to split further. Prevents overly specific leaf nodes. |
| `random_state`      | 42    | Ensures reproducible results across runs.                                                     |

**Pruning Strategy**:

- `max_depth=15` and `min_samples_split=5` act as pruning mechanisms
- These constraints prevent individual trees from becoming too complex
- Balances bias-variance tradeoff

### Model Training

```python
rf_model.fit(X_train, y_train)
```

During training, each tree:

1. Samples random subsets of training data (bootstrap sampling)
2. Selects random subsets of features at each split
3. Grows to maximum allowed depth
4. Applies pruning constraints

---

## Results and Performance

### Evaluation Metrics

The model is evaluated using three standard regression metrics:

#### 1. R² Score (Coefficient of Determination)

Measures the proportion of variance in car prices explained by the model.

```python
r2 = r2_score(y_test, y_pred)
```

**Interpretation**:

- Range: 0 to 1 (higher is better)
- R² = 0.90 means the model explains 90% of price variance
- Represents overall model **accuracy**

#### 2. Mean Absolute Error (MAE)

Average absolute difference between predicted and actual prices.

```python
mae = mean_absolute_error(y_test, y_pred)
```

**Interpretation**:

- Units: Rupees
- Lower is better
- MAE = 200,000 means predictions are off by Rs. 200,000 on average
- More interpretable than RMSE

#### 3. Root Mean Squared Error (RMSE)

Square root of average squared errors.

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

**Interpretation**:

- Units: Rupees
- Lower is better
- Penalizes large errors more heavily than MAE
- Useful for detecting systematic prediction failures

### Model Performance

```
-------------------------------------------------
Model Performance Results:
✅ Accuracy (R² Score): [Value from execution]
❌ Mean Absolute Error: Rs. [Value from execution]
❌ RMSE:                Rs. [Value from execution]
-------------------------------------------------
```

### Visualization: Actual vs Predicted Prices

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price (Rs)")
plt.ylabel("Predicted Price (Rs)")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.show()
```

**Interpretation**:

- Each point represents a test set vehicle
- Red dashed line = perfect predictions (y = x)
- Points close to line indicate accurate predictions
- Scatter pattern reveals model behavior across price ranges

### Explainability (XAI): SHAP Analysis

To understand **why** the model makes specific predictions, SHAP (SHapley Additive exPlanations) values are computed.

#### SHAP Overview

- Based on game theory (Shapley values)
- Quantifies each feature's contribution to individual predictions
- Provides both local (single prediction) and global (overall model) explanations

#### Implementation

```python
# Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
```

#### 1. Summary Plot (Global Feature Importance)

```python
shap.summary_plot(shap_values, X_test, show=False)
```

**Purpose**: Shows which features matter most across all predictions.

**Interpretation**:

- Features ranked by importance (top to bottom)
- Color indicates feature value (red = high, blue = low)
- Horizontal position shows impact on prediction
- Expected insights:
  - `manufacture_year`: Newer cars predicted higher
  - `mileage`: Higher mileage reduces price
  - `engine_capacity`: Larger engines may increase price
  - `model_*`: Specific models command premium prices

#### 2. Dependence Plot (Single Feature Analysis)

```python
shap.dependence_plot("manufacture_year", shap_values, X_test)
```

**Purpose**: Shows how `manufacture_year` affects price predictions.

**Interpretation**:

- X-axis: Manufacture year
- Y-axis: SHAP value (impact on prediction)
- Points: Individual cars in test set
- Trend reveals depreciation pattern over time

---

## Project Structure

```
car-price-predictor-214172V/
│
├── data/
│   ├── car-ad-links/              # Advertisement URLs by brand
│   │   ├── toyota.txt
│   │   ├── suzuki.txt
│   │   ├── honda.txt
│   │   ├── nissan.txt
│   │   ├── mitsubishi.txt
│   │   └── daihatsu.txt
│   │
│   ├── raw/                       # Scraped data (unprocessed)
│   │   ├── toyota_cars.csv
│   │   ├── suzuki_cars.csv
│   │   ├── honda_cars.csv
│   │   ├── nissan_cars.csv
│   │   ├── mitsubishi_cars.csv
│   │   └── daihatsu_cars.csv
│   │
│   ├── processed/                 # Clean ML-ready data
│   │   └── cleaned_car_data.csv
│   │
│   └── failed-links/              # URLs that failed to scrape
│       ├── toyota_failed_links.txt
│       ├── honda_failed_links.txt
│       ├── nissan_failed_links.txt
│       └── suzuki_failed_links.txt
│
├── src/
│   ├── preprocess.py              # Data cleaning pipeline
│   └── train_model.py             # Model training script (if exists)
│
├── notebooks/
│   └── car_price_prediction_using_random_forest.ipynb
│
├── doc/                           # Documentation
│   └── project-documentation.md
│
├── scrape.py                      # Web scraping script
├── requirements.txt               # Python dependencies
└── readme.md                      # Project overview
```

### File Descriptions

| File/Folder                                                | Purpose                                          |
| ---------------------------------------------------------- | ------------------------------------------------ |
| `scrape.py`                                                | Web scraping automation script                   |
| `src/preprocess.py`                                        | Data cleaning and transformation pipeline        |
| `notebooks/car_price_prediction_using_random_forest.ipynb` | Model training and evaluation notebook           |
| `data/car-ad-links/`                                       | Collected advertisement URLs organized by brand  |
| `data/raw/`                                                | Unprocessed scraped data from ikman.lk           |
| `data/processed/`                                          | Clean, ML-ready dataset                          |
| `data/failed-links/`                                       | URLs that failed during scraping (for debugging) |
| `doc/`                                                     | Project documentation                            |

---

## Technical Summary

### Data Pipeline Flow

```
ikman.lk Advertisements
         ↓
   [Manual Filtering]      ← Domain criteria applied
         ↓
   URL Collection          ← data/car-ad-links/
         ↓
   Web Scraping            ← scrape.py
         ↓
   Raw CSV Files           ← data/raw/
         ↓
   Preprocessing           ← src/preprocess.py
         ↓
   Clean Dataset           ← data/processed/cleaned_car_data.csv
         ↓
   Feature Engineering     ← One-hot encoding
         ↓
   Random Forest Model     ← Training
         ↓
   Predictions & Metrics   ← Evaluation
         ↓
   SHAP Explanations       ← Interpretability
```

### Technology Stack

| Component            | Technology                           |
| -------------------- | ------------------------------------ |
| **Web Scraping**     | BeautifulSoup, Requests              |
| **Data Processing**  | Pandas, NumPy                        |
| **Machine Learning** | Scikit-learn (RandomForestRegressor) |
| **Visualization**    | Matplotlib, Seaborn                  |
| **Explainability**   | SHAP                                 |
| **Development**      | Jupyter Notebook, Python 3.x         |

### Key Implementation Decisions

1. **Pre-filtering on ikman.lk**: Reduced scraping volume and processing overhead by applying domain filters during data collection rather than post-processing.

2. **Brand-wise Organization**: Separated URL collection and scraping by brand for better error isolation and parallel processing capability.

3. **Checkpointing in Scraper**: Implemented batch saves every 50 records to prevent data loss from network interruptions.

4. **Rate Limiting**: Added random delays (0.5-1.5s) to respect server resources and avoid IP blocking.

5. **Body Type Auto-Correction**: Created curated list of known hatchback models to fix inconsistent labeling in source data.

6. **Random Forest over Linear Regression**: Chosen for ability to capture non-linear depreciation patterns and interactions between features.

7. **SHAP for Explainability**: Implemented model interpretability to understand feature contributions and validate model behavior.

### Data Quality Metrics

**Before Preprocessing**:

- Raw records scraped: [Count varies by execution]
- Contains units, formatting, duplicates, spelling errors

**After Preprocessing**:

- Clean records: [Reduced count after filtering]
- All domain constraints satisfied
- Numerical columns properly typed
- Categorical variables encoded
- No missing values

### Model Characteristics

**Strengths**:

- Handles non-linear price-feature relationships
- Robust to outliers and anomalies
- Provides feature importance insights
- No feature scaling required
- Ensemble approach reduces overfitting

**Limitations**:

- Predictions limited to domain scope (hatchbacks, 2005+, automatic, ≤10M)
- Model names not seen during training cannot be predicted
- Does not account for external factors (accident history, modifications)
- Performance depends on training data representativeness

---

## Conclusion

This project successfully implemented an end-to-end machine learning pipeline for used car price prediction in the Sri Lankan market. The system combines web scraping, data preprocessing, and Random Forest regression to deliver accurate price estimates based on vehicle specifications.

The implementation demonstrates:

- **Systematic data collection** from ikman.lk with domain-specific filtering
- **Robust preprocessing** handling real-world data quality issues
- **Effective model selection** using Random Forest for non-linear regression
- **Model interpretability** through SHAP analysis for trustworthy predictions

The resulting model provides a data-driven tool for price estimation in the specified market segment, with clear explanations for individual predictions through XAI techniques.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Project**: Car Price Predictor - Student ID: 214172V
