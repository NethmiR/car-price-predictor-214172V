# 🚗 Car Price Prediction System

A machine learning-powered web application for predicting used car prices in the Sri Lankan market. This project uses Random Forest regression to estimate hatchback car prices based on vehicle specifications.

## 📋 Project Overview

This system predicts car prices by analyzing:

- **Mileage** (km)
- **Engine Capacity** (cc)
- **Manufacture Year**
- **Car Model**
- **Fuel Type** (Petrol/Hybrid)

**Target Domain**:

- **Body Type**: Hatchback
- **Transmission**: Automatic
- **Manufacture Years**: After 2005
- **Brands**: Toyota, Suzuki, Honda, Nissan, Mitsubishi, Daihatsu
- **Fuel Type**: Petrol and Hybrid
- **Condition**: Used
- **Price Range**: Up to Rs. 10,000,000

**Model Performance**:

- **R² Score**: 89.38% (model explains 89.38% of price variance)
- **Mean Absolute Error**: Rs. 285,403
- **RMSE**: Rs. 410,993

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 3. Make Predictions

1. Select car brand (Toyota, Suzuki, Honda, etc.)
2. Choose model from the filtered list
3. Enter mileage, engine capacity, and year
4. Select fuel type (Petrol/Hybrid)
5. Click "🔮 Predict Price" to get estimation

---

## 📁 Project Structure

```
car-price-predictor-214172V/
├── app.py                          # Streamlit web application
├── scrape.py                       # Web scraping script (ikman.lk)
├── requirements.txt                # Python dependencies
├── readme.md                       # This file
│
├── data/
│   ├── car-ad-links/              # Advertisement URLs by brand
│   ├── raw/                       # Scraped raw CSV files
│   ├── processed/                 # Cleaned dataset (cleaned_car_data.csv)
│   └── failed-links/              # Failed scraping attempts
│
├── models/
│   ├── random_forest_car_price_model.pkl   # Trained model
│   ├── feature_names.json                  # 46 feature names
│   ├── model_metadata.json                 # Training configuration
│   ├── evaluation_metrics.json             # Performance metrics
│   └── README.md                           # Model usage guide
│
├── notebooks/
│   └── car_price_prediction_using_random_forest.ipynb  # Training notebook
│
├── src/
│   ├── preprocess.py              # Data cleaning pipeline
│   ├── train_model.py             # Model training script
│   └── predict.py                 # Standalone prediction script
│
└── doc/
    ├── project-documentation.md           # Complete technical documentation
    ├── model-training-and-explanation.md  # Model training details
    └── streamlit-app-guide.md            # Web app user guide
```

---

## 🔄 Complete Workflow

### Phase 1: Data Collection

1. **Manual Filtering** on ikman.lk (apply domain filters)
2. **Link Collection** → Save URLs to `data/car-ad-links/{brand}.txt`
3. **Web Scraping** → Run `scrape.py` to extract car details
4. **Output** → Raw CSV files in `data/raw/`

### Phase 2: Data Preprocessing

1. **Consolidation** → Merge all brand CSVs
2. **Cleaning** → Standardize names, remove units, fix errors
3. **Filtering** → Apply domain constraints
4. **Encoding** → One-hot encode fuel type
5. **Output** → `data/processed/cleaned_car_data.csv` (762 samples)

### Phase 3: Model Training

1. **Load Data** → Read cleaned dataset
2. **Feature Engineering** → One-hot encode car models (46 features)
3. **Train-Test Split** → 80/20 split (609 train, 153 test)
4. **Model Training** → Random Forest Regressor
   - 100 estimators
   - max_depth=15 (pruning)
   - min_samples_split=5 (pruning)
5. **Evaluation** → R²=0.8938, MAE=Rs. 285,403
6. **SHAP Analysis** → Feature importance and explainability
7. **Model Persistence** → Save model + metadata to `models/`

### Phase 4: Deployment

1. **Streamlit App** → Interactive web interface
2. **User Input** → Brand, model, specs
3. **Prediction** → Real-time price estimation
4. **Explanation** → Price breakdown and factors

---

## 💻 Usage Examples

### Using the Web App (Recommended)

```bash
streamlit run app.py
```

### Using Python Script

```python
from src.predict import load_model_artifacts, predict_price

# Load model
model, features, metadata = load_model_artifacts()

# Predict price for Toyota Vitz
price = predict_price(
    model=model,
    feature_names=features,
    mileage=47000,
    engine_capacity=990,
    manufacture_year=2018,
    model_name='Vitz',
    fuel_type='Petrol'
)

print(f"Predicted Price: Rs. {price:,.0f}")
```

### Using Jupyter Notebook

See `notebooks/car_price_prediction_using_random_forest.ipynb` for:

- Complete training pipeline
- Model evaluation
- SHAP explainability analysis
- Sample predictions

---

## 🛠️ Technologies Used

**Web Scraping**:

- `beautifulsoup4` - HTML parsing
- `requests` - HTTP requests
- `tqdm` - Progress bars

**Data Processing**:

- `pandas` - Data manipulation
- `numpy` - Numerical operations

**Machine Learning**:

- `scikit-learn` - Random Forest Regressor
- `joblib` - Model serialization

**Visualization & Explainability**:

- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `shap` - Model explainability (SHapley Additive exPlanations)

**Web Application**:

- `streamlit` - Interactive web interface

---

## 📊 Model Details

**Algorithm**: Random Forest Regressor

**Why Random Forest?**

- ✅ Handles non-linear relationships (car depreciation is non-linear)
- ✅ Robust to outliers (unusual prices, rare editions)
- ✅ No feature scaling required (different units: km, cc, year)
- ✅ Built-in feature importance
- ✅ Ensemble learning reduces overfitting

**Hyperparameters**:

- `n_estimators=100` - Build 100 decision trees
- `max_depth=15` - Limit tree depth (prevents overfitting)
- `min_samples_split=5` - Minimum samples to split node
- `random_state=42` - Reproducibility

**Features** (46 total):

- **Continuous** (4): mileage, engine_capacity, manufacture_year, fuel_type_Petrol
- **One-Hot Encoded** (42): model_Alto, model_Aqua, ..., model_Yaris

---

## 📖 Documentation

Comprehensive documentation is available in the `doc/` directory:

1. **[Project Documentation](doc/project-documentation.md)**
   - Complete pipeline overview
   - Data collection methodology
   - Web scraping implementation
   - Preprocessing pipeline (7 stages)
   - Model development process

2. **[Model Training & Explanation](doc/model-training-and-explanation.md)**
   - Input data structure
   - Feature engineering details
   - Model architecture justification
   - Training process
   - Evaluation metrics
   - SHAP explainability analysis
   - Technical specifications

3. **[Streamlit App Guide](doc/streamlit-app-guide.md)**
   - Installation & setup
   - User guide (step-by-step)
   - Understanding predictions
   - Technical architecture
   - Troubleshooting
   - Deployment options

---

## 🔍 Data Source

**Platform**: [ikman.lk](https://ikman.lk) (Sri Lanka's largest marketplace)

**Collection Method**:

- Manual filtering on website (apply domain constraints)
- URL collection to brand-specific text files
- Automated scraping with `scrape.py`
- Checkpointing every 50 records (prevents data loss)
- Rate limiting (0.5-1.5s delays to avoid blocking)

**Dataset Statistics**:

- **Total Samples**: 762 car listings
- **Brands**: 6 (Toyota, Suzuki, Honda, Nissan, Mitsubishi, Daihatsu)
- **Models**: 47 unique models (A-Star to Yaris)
- **Price Range**: Rs. 2,050,000 - Rs. 9,950,000
- **Year Range**: 2005 - 2025
- **Mileage Range**: 0 - 250,000 km

---

## 🎯 Key Features

### Web Application

- ✅ Interactive brand and model selection
- ✅ Dynamic model filtering by brand
- ✅ Input validation with warnings
- ✅ Real-time price predictions
- ✅ Price range estimation (±10%)
- ✅ Detailed factor analysis (age, mileage impact)
- ✅ Model performance metrics display
- ✅ Responsive design

### Model Capabilities

- ✅ 89.38% accuracy (R² score)
- ✅ Average error ±Rs. 285K
- ✅ SHAP explainability (understand feature impact)
- ✅ Handles 47 different car models
- ✅ Accounts for fuel type (Petrol vs. Hybrid)

---

## ⚠️ Limitations

1. **Domain-Specific**:
   - Only hatchbacks with automatic transmission
   - Limited to Petrol/Hybrid fuel types
   - Trained on Sri Lankan market data

2. **Excluded Factors**:
   - Vehicle condition/damage
   - Service history
   - Modifications/upgrades
   - Seller reputation
   - Location-based pricing
   - Current market demand

3. **Data Freshness**:
   - Trained on historical ikman.lk data
   - May not reflect current market fluctuations
   - Requires periodic retraining

---

## 🚦 Getting Started (Detailed)

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Guide

#### 1. Clone/Download Project

```bash
cd car-price-predictor-214172V
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Verify Model Files

Check that these files exist in `models/`:

- `random_forest_car_price_model.pkl`
- `feature_names.json`
- `model_metadata.json`
- `evaluation_metrics.json`

If missing, run the training notebook:

```bash
jupyter notebook notebooks/car_price_prediction_using_random_forest.ipynb
```

#### 5. Launch Web Application

```bash
streamlit run app.py
```

#### 6. Open Browser

Navigate to `http://localhost:8501`

---

## 🧪 Running Individual Components

### Web Scraping (Optional - Data Already Collected)

```bash
python scrape.py
```

- Reads URLs from `data/car-ad-links/*.txt`
- Outputs to `data/raw/{brand}_cars.csv`
- Failed links saved to `data/failed-links/`

### Data Preprocessing

```bash
python src/preprocess.py
```

- Reads from `data/raw/*.csv`
- Outputs to `data/processed/cleaned_car_data.csv`

### Model Training (Jupyter Notebook)

```bash
jupyter notebook notebooks/car_price_prediction_using_random_forest.ipynb
```

Run all cells to:

- Load and explore data
- Engineer features
- Train Random Forest model
- Evaluate performance
- Generate SHAP explanations
- Save model artifacts

### Standalone Prediction Script

```bash
python src/predict.py
```

Demonstrates loading model and making predictions programmatically.

---

## 🤝 Contributing

To improve this project:

1. **More Data**: Add more brands, body types, or features
2. **Feature Engineering**: Add location, color, seller type
3. **Model Improvement**: Try Gradient Boosting, XGBoost, or Neural Networks
4. **Web App**: Add charts, comparison tools, or export features
5. **Real-time Data**: Integrate live market data API

---

## 📞 Support

For issues or questions:

1. Check [Streamlit App Guide](doc/streamlit-app-guide.md) for troubleshooting
2. Review [Project Documentation](doc/project-documentation.md) for technical details
3. Verify all model files are present in `models/` directory

---

## 📜 License

This project is for educational purposes as part of a Machine Learning coursework assignment.

**Data Source**: ikman.lk  
**Framework**: Streamlit, scikit-learn  
**Model**: Random Forest Regressor

---

## 🎓 Academic Context

**Course**: Machine Learning  
**Assignment**: Car Price Prediction with Explainability  
**Requirements**:

- ✅ Data collection and preprocessing
- ✅ Model training with hyperparameter tuning
- ✅ Pruning techniques (max_depth, min_samples_split)
- ✅ Explainability (SHAP analysis)
- ✅ Front-end integration (Streamlit web app)
- ✅ Comprehensive documentation

---

## 🏆 Project Highlights

- **762 samples** collected and preprocessed
- **89.38% R² score** achieved
- **46 features** engineered
- **SHAP explainability** implemented
- **Interactive web app** deployed
- **Comprehensive documentation** (3 detailed guides)

---

**Built with ❤️ for Machine Learning Course | 2026**
