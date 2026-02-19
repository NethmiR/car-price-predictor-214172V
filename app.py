import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# LOAD MODEL ARTIFACTS (CACHED)
# ==========================================
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and related artifacts"""
    try:
        model = joblib.load('models/random_forest_car_price_model.pkl')
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        with open('models/evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        return model, feature_names, metadata, metrics
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None, None

# ==========================================
# EXTRACT BRAND-MODEL MAPPING
# ==========================================
def extract_brand_models(feature_names):
    """Extract available models from feature names and map to brands"""
    model_features = [f.replace('model_', '') for f in feature_names if f.startswith('model_')]
    
    # Add the reference category (A-Star) which was dropped during encoding
    all_models = ['A-Star'] + model_features
    
    # Brand to Model mapping (based on actual car manufacturers)
    brand_model_mapping = {
        'Toyota': ['Aqua', 'Glanza', 'Passo', 'Pixis', 'Roomy', 'Vitz', 'Yaris'],
        'Suzuki': ['A-Star', 'Alto', 'Baleno', 'Celerio', 'Wagon R'],
        'Honda': ['Fit'],
        'Nissan': ['Dayz', 'March', 'Moco', 'Roox'],
        'Mitsubishi': ['Miraa'],
        'Daihatsu': ['Move', 'Taft']
    }
    
    # Filter only models that exist in our dataset
    filtered_mapping = {}
    for brand, models in brand_model_mapping.items():
        available = [m for m in models if m in all_models]
        if available:
            filtered_mapping[brand] = sorted(available)
    
    return filtered_mapping

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================
def create_input_dataframe(mileage, engine_capacity, manufacture_year, 
                          model_name, fuel_type, feature_names):
    """
    Create a properly formatted input DataFrame for prediction
    
    Parameters:
    - mileage: int - odometer reading in km
    - engine_capacity: int - engine size in cc
    - manufacture_year: int - year of manufacture
    - model_name: str - car model name
    - fuel_type: str - 'Petrol' or 'Hybrid'
    - feature_names: list - list of all 46 feature names
    
    Returns:
    - DataFrame with 1 row and 46 columns in correct order
    """
    # Initialize all features to 0
    input_dict = {feature: 0 for feature in feature_names}
    
    # Set continuous features
    input_dict['mileage'] = int(mileage)
    input_dict['engine_capacity'] = int(engine_capacity)
    input_dict['manufacture_year'] = int(manufacture_year)
    
    # Set fuel type (binary)
    input_dict['fuel_type_Petrol'] = 1 if fuel_type == 'Petrol' else 0
    
    # Set model one-hot encoding
    # Note: A-Star (reference category) is not in features, so it remains all 0s
    model_column = f'model_{model_name}'
    if model_column in input_dict:
        input_dict[model_column] = 1
    
    # Create DataFrame with exact feature order
    return pd.DataFrame([input_dict])

# ==========================================
# DISPLAY PREDICTION RESULTS
# ==========================================
def display_results(prediction, mileage, engine_capacity, manufacture_year, 
                   model_name, fuel_type, brand):
    """Display prediction results with detailed breakdown"""
    
    # Main prediction
    st.success("### 💰 Predicted Price")
    st.markdown(f"## Rs. {prediction:,.0f}")
    
    # Calculate vehicle age
    current_year = datetime.now().year
    age = current_year - manufacture_year
    
    # Price range (±10%)
    lower_bound = prediction * 0.9
    upper_bound = prediction * 1.1
    
    st.markdown("---")
    st.subheader("📊 Price Estimation Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Vehicle Specifications")
        st.markdown(f"""
        - **Brand**: {brand}
        - **Model**: {model_name}
        - **Year**: {manufacture_year} ({age} years old)
        - **Mileage**: {mileage:,} km
        - **Engine**: {engine_capacity:,} cc
        - **Fuel Type**: {fuel_type}
        """)
    
    with col2:
        st.markdown("#### Price Factors")
        
        # Age assessment
        if age < 3:
            age_desc = "Very New"
            age_impact = "⬆️ Increases value"
        elif age < 6:
            age_desc = "Relatively New"
            age_impact = "➡️ Moderate value"
        elif age < 10:
            age_desc = "Moderate Age"
            age_impact = "⬇️ Decreases value slightly"
        else:
            age_desc = "Older Vehicle"
            age_impact = "⬇️⬇️ Decreases value"
        
        # Mileage assessment
        if mileage < 30000:
            mileage_desc = "Very Low"
            mileage_impact = "⬆️ Increases value"
        elif mileage < 60000:
            mileage_desc = "Low Mileage"
            mileage_impact = "➡️ Good value"
        elif mileage < 100000:
            mileage_desc = "Average Mileage"
            mileage_impact = "➡️ Normal value"
        else:
            mileage_desc = "High Mileage"
            mileage_impact = "⬇️ Decreases value"
        
        st.markdown(f"""
        - **Age Factor**: {age_desc}  
          *{age_impact}*
        - **Mileage Factor**: {mileage_desc}  
          *{mileage_impact}*
        - **Fuel Efficiency**: {fuel_type}  
          *{'⬆️ Eco-friendly' if fuel_type == 'Hybrid' else '➡️ Standard'}*
        """)
    
    # Price range
    st.markdown("---")
    st.info(f"""
    **💡 Expected Price Range**: Rs. {lower_bound:,.0f} - Rs. {upper_bound:,.0f}
    
    *This estimation is based on {manufacture_year} {brand} {model_name} with {mileage:,} km 
    mileage and {engine_capacity} cc engine capacity. The actual selling price may vary based on 
    vehicle condition, service history, modifications, market demand, and negotiation.*
    """)

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # Header
    st.title("🚗 Car Price Prediction System")
    st.markdown("""
    ### Predict the price of used hatchback cars in Sri Lanka
    This AI-powered system uses a Random Forest machine learning model trained on real market data 
    to predict car prices based on vehicle specifications.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, feature_names, metadata, metrics = load_model_artifacts()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure model files exist in the 'models/' directory.")
        return
    
    # Show model info in sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown(f"""
        **Model Type**: Random Forest Regressor  
        **Training Date**: {metadata.get('trained_date', 'N/A')}  
        **Training Samples**: {metadata.get('training_samples', 'N/A')}  
        **Features**: {metadata.get('n_features', 'N/A')}  
        
        **Performance Metrics**:
        - **Accuracy (R²)**: {metrics.get('r2_percentage', 0):.2f}%
        - **Avg Error**: Rs. {metrics.get('mean_absolute_error', 0):,.0f}
        
        ---
        
        **Domain Scope**:
        - Body Type: Hatchback
        - Transmission: Automatic
        - Fuel: Petrol/Hybrid
        - Year: 2005+
        - Max Price: Rs. 10M
        """)
    
    st.success("✅ Model loaded successfully!")
    
    # Extract brand-model mapping
    brand_models = extract_brand_models(feature_names)
    
    # Input Form
    st.markdown("---")
    st.subheader("📝 Enter Vehicle Details")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Brand Selection
        brand = st.selectbox(
            "🏢 Select Brand",
            options=list(brand_models.keys()),
            help="Choose the car manufacturer"
        )
        
        # 2. Model Selection (filtered by brand)
        model_name = st.selectbox(
            "🚙 Select Model",
            options=brand_models.get(brand, []),
            help="Choose the car model based on selected brand"
        )
        
        # 3. Mileage Input
        mileage = st.number_input(
            "📏 Mileage (km)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=5000,
            help="Total distance traveled by the vehicle (odometer reading)"
        )
        
        # Mileage validation warning
        if mileage > 250000:
            st.warning("⚠️ Very high mileage detected. This may affect prediction accuracy.")
    
    with col2:
        # 4. Engine Capacity Input
        engine_capacity = st.number_input(
            "⚙️ Engine Capacity (cc)",
            min_value=600,
            max_value=7000,
            value=1000,
            step=50,
            help="Engine displacement in cubic centimeters"
        )
        
        # Engine validation warning
        if engine_capacity < 600 or engine_capacity > 2000:
            st.warning("⚠️ Engine capacity outside typical range (600-2000 cc) for hatchbacks.")
        
        # 5. Year of Manufacture
        manufacture_year = st.number_input(
            "📅 Year of Manufacture",
            min_value=2005,
            max_value=datetime.now().year,
            value=2018,
            step=1,
            help="Year the vehicle was manufactured"
        )
        
        # 6. Fuel Type
        fuel_type = st.radio(
            "⛽ Fuel Type",
            options=['Petrol', 'Hybrid'],
            help="Select the fuel type of the vehicle"
        )
    
    # Prediction Button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    # Make Prediction
    if predict_button:
        with st.spinner("Calculating price prediction..."):
            try:
                # Create input DataFrame
                input_data = create_input_dataframe(
                    mileage=mileage,
                    engine_capacity=engine_capacity,
                    manufacture_year=manufacture_year,
                    model_name=model_name,
                    fuel_type=fuel_type,
                    feature_names=feature_names
                )
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Validate prediction
                if prediction < 0:
                    st.error("❌ Model returned negative price. Please check input values.")
                elif prediction > 20000000:
                    st.warning("⚠️ Predicted price exceeds typical market range. Please verify inputs.")
                    display_results(prediction, mileage, engine_capacity, manufacture_year, 
                                  model_name, fuel_type, brand)
                else:
                    # Display results
                    display_results(prediction, mileage, engine_capacity, manufacture_year, 
                                  model_name, fuel_type, brand)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please check your inputs and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        <p>🎓 Car Price Prediction ML Project | Built with Streamlit & scikit-learn</p>
        <p>Model trained on Sri Lankan used car market data (ikman.lk)</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# RUN APPLICATION
# ==========================================
if __name__ == "__main__":
    main()
