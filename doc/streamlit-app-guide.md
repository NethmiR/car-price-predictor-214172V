# Streamlit Car Price Prediction Web Application

## Overview

This document provides comprehensive guidance for using the Streamlit web application that integrates the trained Random Forest car price prediction model.

**Application File**: `app.py`

**Purpose**: Allow users to input vehicle specifications through an interactive web interface and receive real-time price predictions with detailed explanations.

---

## Features

### 🎯 Core Functionality

1. **Interactive Input Interface**
   - Brand selection dropdown (Toyota, Suzuki, Honda, Nissan, Mitsubishi, Daihatsu)
   - Dynamic model selection (filtered by selected brand)
   - Validated input fields for mileage, engine capacity, and year
   - Fuel type selection (Petrol/Hybrid)

2. **Real-Time Predictions**
   - Instant price prediction using trained Random Forest model
   - Price range estimation (±10% confidence interval)
   - Detailed price breakdown and factor analysis

3. **Model Transparency**
   - Display model performance metrics (R² score, MAE)
   - Show training information and dataset size
   - Explain domain constraints

4. **User-Friendly Design**
   - Clean, responsive layout
   - Clear instructions and help tooltips
   - Visual warnings for unusual input values
   - Formatted currency display (Sri Lankan Rupees)

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- All dependencies from `requirements.txt`
- Trained model artifacts in `models/` directory

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- `streamlit>=1.32.0` - Web application framework
- `joblib>=1.3.0` - Model loading
- `pandas`, `numpy` - Data processing
- `scikit-learn>=1.3.0` - ML framework (required for prediction)

### Step 2: Verify Model Files

Ensure the following files exist in the `models/` directory:

```
models/
├── random_forest_car_price_model.pkl    (trained model)
├── feature_names.json                   (46 feature names)
├── model_metadata.json                  (training configuration)
└── evaluation_metrics.json              (performance metrics)
```

If these files are missing, run the training notebook first:

- `notebooks/car_price_prediction_using_random_forest.ipynb`

### Step 3: Launch Application

From the project root directory, run:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

---

## User Guide

### Using the Application

#### Step 1: Select Brand

Choose the car manufacturer from the dropdown:

- Toyota
- Suzuki
- Honda
- Nissan
- Mitsubishi
- Daihatsu

#### Step 2: Select Model

The second dropdown will dynamically populate with models available for the selected brand.

**Brand-Model Mapping**:

- **Toyota**: Aqua, Glanza, Passo, Pixis, Roomy, Vitz, Yaris
- **Suzuki**: A-Star, Alto, Baleno, Celerio, Wagon R
- **Honda**: Fit
- **Nissan**: Dayz, March, Moco, Roox
- **Mitsubishi**: Miraa
- **Daihatsu**: Move, Taft

#### Step 3: Enter Vehicle Specifications

**Mileage (km)**:

- Enter the total distance traveled (odometer reading)
- Valid range: 0 - 500,000 km
- Step: 5,000 km
- Warning displayed if mileage > 250,000 km

**Engine Capacity (cc)**:

- Enter engine displacement in cubic centimeters
- Valid range: 600 - 7,000 cc
- Step: 50 cc
- Typical hatchback range: 600 - 2,000 cc

**Year of Manufacture**:

- Year the vehicle was manufactured
- Valid range: 2005 - 2026
- Model trained on vehicles from 2005 onwards

**Fuel Type**:

- Petrol: Traditional gasoline engine
- Hybrid: Petrol-electric hybrid engine

#### Step 4: Get Prediction

Click the **"🔮 Predict Price"** button to generate prediction.

---

## Understanding Results

### Predicted Price

The main prediction displays the estimated market value in Sri Lankan Rupees:

```
💰 Predicted Price
Rs. 7,245,000
```

### Price Estimation Breakdown

#### Vehicle Specifications Panel

Shows the input details:

- Brand and Model
- Year and Age (calculated)
- Mileage and Engine Capacity
- Fuel Type

#### Price Factors Panel

Analyzes how different factors influence the price:

**Age Factor**:

- Very New (< 3 years): ⬆️ Increases value
- Relatively New (3-6 years): ➡️ Moderate value
- Moderate Age (6-10 years): ⬇️ Decreases value slightly
- Older Vehicle (> 10 years): ⬇️⬇️ Decreases value

**Mileage Factor**:

- Very Low (< 30,000 km): ⬆️ Increases value
- Low (30,000-60,000 km): ➡️ Good value
- Average (60,000-100,000 km): ➡️ Normal value
- High (> 100,000 km): ⬇️ Decreases value

**Fuel Efficiency**:

- Hybrid: ⬆️ Eco-friendly (typically higher value)
- Petrol: ➡️ Standard

### Expected Price Range

The application provides a confidence interval (±10%):

```
💡 Expected Price Range: Rs. 6,520,500 - Rs. 7,969,500
```

This range accounts for:

- Vehicle condition variations
- Service history differences
- Market demand fluctuations
- Negotiation factors

---

## Technical Architecture

### Data Flow

```
User Input → Input Validation → Feature Engineering → Model Prediction → Result Display
```

#### 1. Input Collection

- Brand/Model dropdowns (categorical selection)
- Number inputs with range validation (mileage, engine, year)
- Radio button (fuel type)

#### 2. Input Validation

- Range checks (min/max values)
- Data type conversion (string → int)
- Warning messages for unusual values

#### 3. Feature Engineering

- **Create 46-feature input vector**:
  - 4 continuous features: `mileage`, `engine_capacity`, `manufacture_year`, `fuel_type_Petrol`
  - 42 one-hot encoded model features: `model_Alto`, `model_Aqua`, ..., `model_Yaris`
- **One-Hot Encoding Logic**:
  - Initialize all features to 0
  - Set selected model's feature to 1
  - Reference category (A-Star) remains all 0s
- **Feature Order**: Must match training order from `feature_names.json`

#### 4. Model Prediction

- Load trained Random Forest model using `joblib`
- Pass 1×46 DataFrame to `model.predict()`
- Return single price prediction

#### 5. Result Presentation

- Format price with thousand separators
- Calculate price range (±10%)
- Analyze age and mileage factors
- Generate descriptive explanation

### Code Structure

```python
app.py
├── load_model_artifacts()          # Load model + metadata (cached)
├── extract_brand_models()          # Build brand-model mapping
├── create_input_dataframe()        # Feature engineering
├── display_results()               # Format and show predictions
└── main()                          # Application entry point
```

### Key Functions

#### `load_model_artifacts()`

```python
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('models/random_forest_car_price_model.pkl')
    # ... load JSON files
    return model, feature_names, metadata, metrics
```

- **Caching**: Uses `@st.cache_resource` to load model only once
- **Error Handling**: Returns `None` values if loading fails

#### `extract_brand_models(feature_names)`

```python
def extract_brand_models(feature_names):
    model_features = [f.replace('model_', '') for f in feature_names if f.startswith('model_')]
    all_models = ['A-Star'] + model_features  # Add dropped reference
    # ... create brand mapping
    return filtered_mapping
```

- Dynamically extracts available models from feature names
- Maps models to correct brands
- Includes A-Star (dropped reference category)

#### `create_input_dataframe()`

```python
def create_input_dataframe(mileage, engine_capacity, manufacture_year,
                          model_name, fuel_type, feature_names):
    input_dict = {feature: 0 for feature in feature_names}  # Initialize
    input_dict['mileage'] = int(mileage)                   # Set continuous
    # ... set other features
    input_dict[f'model_{model_name}'] = 1                  # One-hot encode
    return pd.DataFrame([input_dict])                      # Return 1x46 DF
```

- Critical for correct predictions
- Ensures feature order matches training
- Handles one-hot encoding properly

---

## Model Information (Sidebar)

The application displays model metadata in the sidebar:

### Performance Metrics

- **Accuracy (R²)**: 89.38%
  - Model explains 89.38% of price variance
- **Average Error**: Rs. 285,403
  - Typical prediction deviation

### Training Information

- **Training Date**: When model was trained
- **Training Samples**: 609 samples (80% of 762 total)
- **Features**: 46 features after encoding

### Domain Scope

- **Body Type**: Hatchback only
- **Transmission**: Automatic only
- **Fuel Type**: Petrol or Hybrid
- **Year Range**: 2005 onwards
- **Price Cap**: Rs. 10,000,000

---

## Troubleshooting

### Common Issues

#### 1. "Failed to load model"

**Cause**: Model artifacts missing or incorrect path

**Solution**:

- Verify files exist in `models/` directory
- Run training notebook to generate model files
- Check file permissions

#### 2. "Prediction failed"

**Cause**: Input data format mismatch

**Solution**:

- Check all inputs are within valid ranges
- Ensure model file is not corrupted
- Verify feature_names.json exists

#### 3. "Model returned negative price"

**Cause**: Unusual input combination outside training distribution

**Solution**:

- Verify inputs are reasonable (especially year and mileage)
- Check if model is appropriate for the vehicle type
- Consider re-training with more diverse data

#### 4. Application doesn't start

**Cause**: Streamlit not installed or incorrect command

**Solution**:

```bash
pip install streamlit
streamlit run app.py
```

### Validation Warnings

The application shows warnings for:

- **Mileage > 250,000 km**: Very high mileage
- **Engine < 600 cc or > 2,000 cc**: Outside typical hatchback range
- **Price > Rs. 20,000,000**: Prediction exceeds normal market range

These warnings indicate potential input errors or edge cases where model accuracy may be reduced.

---

## Deployment Options

### Local Development

```bash
streamlit run app.py
```

Accessible at: `http://localhost:8501`

### Network Access

```bash
streamlit run app.py --server.address 0.0.0.0
```

Accessible from other devices on the same network.

### Cloud Deployment

#### Streamlit Community Cloud (Free)

1. Push code to GitHub repository
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository and deploy
4. Application gets public URL

#### Other Platforms

- **Heroku**: Use `setup.sh` and `Procfile`
- **AWS EC2**: Run on virtual machine
- **Google Cloud Run**: Container deployment
- **Azure Web Apps**: Platform-as-a-Service

---

## Best Practices

### For Users

1. **Verify Input Accuracy**: Double-check mileage, year, and engine capacity
2. **Use Price Range**: Don't rely solely on exact prediction
3. **Consider Additional Factors**: Model doesn't account for:
   - Service history and maintenance records
   - Accident history or body damage
   - Interior/exterior condition
   - Market demand fluctuations
   - Modifications or upgrades
   - Location-based price variations

### For Developers

1. **Model Updates**: Retrain periodically with new market data
2. **Feature Addition**: Consider adding more features (color, location, seller type)
3. **Error Logging**: Implement logging for failed predictions
4. **User Analytics**: Track popular models and input patterns
5. **A/B Testing**: Test different price range percentages

---

## Future Enhancements

### Planned Features

- [ ] Multiple model comparison (Random Forest vs. Gradient Boosting)
- [ ] SHAP explanation visualizations for individual predictions
- [ ] Historical price trends for selected model
- [ ] Batch prediction (CSV upload)
- [ ] Export prediction report (PDF)
- [ ] User feedback collection
- [ ] Save prediction history
- [ ] Add image upload for condition assessment
- [ ] Integration with live market data

### Advanced Analytics

- [ ] Feature importance visualization per prediction
- [ ] Similar car comparison
- [ ] Price depreciation calculator
- [ ] Market trend analysis

---

## Support & Feedback

### Reporting Issues

If you encounter problems:

1. Check troubleshooting section
2. Verify all model files are present
3. Review input values for validity
4. Check console logs for error messages

### Model Limitations

- Trained on data from ikman.lk (Sri Lankan market)
- Limited to hatchbacks with automatic transmission
- Best accuracy for vehicles within training distribution
- Does not account for vehicle condition or modifications

---

## Technical Specifications

**Framework**: Streamlit 1.32.0+  
**ML Library**: scikit-learn 1.3.0+  
**Python Version**: 3.8+  
**Model Type**: Random Forest Regressor  
**Input Features**: 46 (4 continuous + 42 one-hot encoded)  
**Output**: Single continuous value (price in Rs)

**Browser Compatibility**: Chrome, Firefox, Safari, Edge (latest versions)  
**Responsive Design**: Desktop and tablet optimized

---

## License & Attribution

This application is part of a machine learning project for car price prediction in the Sri Lankan used car market.

**Data Source**: ikman.lk  
**Model Training**: Random Forest Regressor (scikit-learn)  
**Framework**: Streamlit  
**Developer**: ML Project Team

---

## Conclusion

The Streamlit Car Price Prediction Application provides an intuitive interface for estimating used hatchback car prices in Sri Lanka. By combining machine learning with user-friendly design, it enables users to make data-driven decisions when buying or selling vehicles.

For technical documentation on model training and data processing, refer to:

- [Project Documentation](project-documentation.md)
- [Model Training Documentation](model-training-and-explanation.md)
