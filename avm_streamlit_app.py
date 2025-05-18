import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import base64
import requests
import io

# Set page configuration
st.set_page_config(
    page_title="Real Estate AVM Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1e88e5;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .feature-importance {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f1f8e9;
        border-radius: 5px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .profit-positive {
        color: #4caf50;
        font-weight: bold;
    }
    .profit-negative {
        color: #f44336;
        font-weight: bold;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Function to load model and features
@st.cache_resource
def load_model():
    model_path = 'models/avm_random_forest.pkl'
    feature_path = 'models/model_features.pkl'
    
    # Check if models exist in the path
    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # For demo purposes, create a simple model if the real one isn't available
        # In a real scenario, you'd want to handle this differently
        from sklearn.ensemble import RandomForestRegressor
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        dummy_model.fit(np.array([[1, 2, 3]]), np.array([100000]))
        joblib.dump(dummy_model, model_path)
        
        dummy_features = ['sqft', 'beds', 'baths']
        with open(feature_path, 'wb') as f:
            pickle.dump(dummy_features, f)
    
    # Load the model and features
    model = joblib.load(model_path)
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    
    return model, features

# Load model and features
model, model_features = load_model()

# Function to create default properties
def create_default_properties():
    # Create sample properties based on realistic data
    return pd.DataFrame({
        'property_id': [1001, 1002, 1003, 1004, 1005],
        'address': [
            '123 Main St, Los Angeles, CA', 
            '456 Oak Ave, San Diego, CA', 
            '789 Pine Rd, Irvine, CA', 
            '101 Sunset Blvd, Malibu, CA', 
            '202 Lake View Dr, Pasadena, CA'
        ],
        'sqft': [1800, 2200, 1500, 3500, 2000],
        'beds': [3, 4, 2, 5, 3],
        'baths': [2.5, 3, 2, 4.5, 2],
        'lot_size': [7500, 8500, 5500, 12000, 7000],
        'year_built': [1985, 2002, 1965, 2015, 1978],
        'property_type': ['SFR', 'SFR', 'SFR', 'SFR', 'SFR'],
        'latitude': [34.0522, 32.7157, 33.6846, 34.0259, 34.1478],
        'longitude': [-118.2437, -117.1611, -117.8265, -118.7798, -118.1445],
        'sale_date': ['2023-05-15', '2023-06-20', '2023-04-10', '2023-07-05', '2023-03-25'],
        'sale_price': [750000, 950000, 620000, 2200000, 850000]
    })

# Create default properties
default_properties = create_default_properties()

# Function to prepare data for prediction
def prepare_for_prediction(property_data, features):
    # Create a dictionary to store feature values
    feature_dict = {}
    
    # Set base features from property data
    base_features = {
        'sqft': property_data.get('sqft', 0),
        'beds': property_data.get('beds', 0),
        'baths': property_data.get('baths', 0),
        'lot_size': property_data.get('lot_size', 0),
        'year_built': property_data.get('year_built', 0),
    }
    
    # Calculate derived features based on our feature engineering process
    # Current year for age calculation
    current_year = datetime.now().year
    property_age = current_year - base_features['year_built']
    
    # Calculate season from month (for season_price_factor)
    current_month = datetime.now().month
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Fall',
        10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    season = season_map[current_month]
    
    season_factor_map = {
        'Spring': 1.05,  # 5% premium in spring
        'Summer': 1.03,  # 3% premium in summer
        'Fall': 0.98,    # 2% discount in fall
        'Winter': 0.94   # 6% discount in winter
    }
    season_price_factor = season_factor_map[season]
    
    # Set neighborhood averages (would normally come from a database)
    # For demo, use fixed values based on neighborhood
    neighborhood_avg_sqft = 2000
    neighborhood_avg_price = 800000
    neighborhood_avg_ppsqft = 400
    
    # Create a dictionary for engineered features
    engineered_features = {
        'property_age': property_age,
        'age_log': np.log1p(property_age + 1),  # Add 1 to handle age=0
        'age_squared': property_age ** 2,
        'bed_bath_ratio': base_features['beds'] / max(base_features['baths'], 1),  # Prevent division by zero
        'living_space_ratio': (base_features['sqft'] / max(base_features['lot_size'], 1)) * 100,
        'relative_size': base_features['sqft'] / neighborhood_avg_sqft,
        'price_per_sqft': property_data.get('sale_price', neighborhood_avg_price) / max(base_features['sqft'], 1),
        'ppsqft_log': np.log1p(property_data.get('sale_price', neighborhood_avg_price) / max(base_features['sqft'], 1)),
        'neighborhood_avg_sqft': neighborhood_avg_sqft,
        'neighborhood_avg_price': neighborhood_avg_price,
        'neighborhood_avg_ppsqft': neighborhood_avg_ppsqft,
        'luxury_score': 50,  # Default value, would be calculated based on property features
        'season_price_factor': season_price_factor,
        'days_since_start': 365,  # Placeholder
    }
    
    # Combine base and engineered features
    all_features = {**base_features, **engineered_features}
    
    # Calculate neighborhood tiers and additional features
    neighborhood_price_tier = 3  # Default mid-tier
    all_features['neighborhood_price_tier'] = neighborhood_price_tier
    all_features['prestige_score'] = (6 - neighborhood_price_tier) * 20
    
    # Add text-derived features (would come from NLP in a real system)
    text_features = {
        'has_renovated': 0,
        'has_condition': 0,
        'has_outdoor': 0,
    }
    all_features.update(text_features)
    
    # Create geo_cluster features (simplified)
    geo_cluster = 2  # Default cluster
    
    # Set cluster-specific features to 0
    for i in range(5):
        all_features[f'sqft_in_cluster_{i}.0'] = 0
        all_features[f'luxury_in_tier_{i+1}'] = 0
    
    # Set the specific cluster feature to sqft
    all_features[f'sqft_in_cluster_{geo_cluster}.0'] = base_features['sqft']
    all_features[f'luxury_in_tier_{neighborhood_price_tier}'] = all_features['luxury_score']
    
    # Create bedroom-specific features
    for i in range(1, 7):
        all_features[f'beds_{i}'] = 1 if base_features['beds'] == i else 0
    
    # Create bathroom-specific features
    for i in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
        all_features[f'baths_{i}'] = 1 if base_features['baths'] == i else 0
    
    # Add renovation interaction feature for mid-century homes
    if 1950 <= base_features['year_built'] <= 1980:
        era = 'mid-century'
    elif base_features['year_built'] < 1950:
        era = 'historic'
    elif base_features['year_built'] <= 2000:
        era = 'contemporary'
    else:
        era = 'modern'
    
    all_features[f'renovated_{era}'] = all_features['has_renovated']
    
    # Prepare final feature array ensuring all model features are present
    final_features = {}
    for feature in features:
        if feature in all_features:
            final_features[feature] = all_features[feature]
        else:
            # If a feature is missing, set to 0 (could be improved with better defaults)
            final_features[feature] = 0
    
    return pd.DataFrame([final_features])

# Function to predict property value
def predict_property_value(property_data, model, features, log_transform=True):
    """
    Predict property value based on the model and features.
    
    Parameters:
    -----------
    property_data : dict
        Dictionary containing property features
    model : trained model
        The model to use for prediction
    features : list
        List of feature names expected by the model
    log_transform : bool
        Whether the model was trained on log-transformed target
        
    Returns:
    --------
    float
        Predicted property value
    """
    # Prepare data for prediction
    X = prepare_for_prediction(property_data, features)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # If model was trained on log-transformed target, convert back to original scale
    if log_transform:
        prediction = np.expm1(prediction)
    
    return prediction

# Function to forecast future values
def forecast_future_value(property_data, months_ahead=1):
    """
    Forecast future property value based on market trends.
    
    Parameters:
    -----------
    property_data : dict
        Dictionary containing property features
    months_ahead : int
        Number of months to forecast ahead
        
    Returns:
    --------
    float
        Forecasted property value
    """
    # Get current prediction
    current_value = predict_property_value(property_data, model, model_features)
    
    # For demonstration, use simple growth rate based on month
    # In a real implementation, this would use a more sophisticated time series model
    monthly_growth_rates = {
        1: 0.005,  # 0.5% monthly growth (6% annual)
        2: 0.010,  # 1.0% for 2 months ahead
        3: 0.016,  # 1.6% for 3 months ahead
        4: 0.022,  # 2.2% for 4 months ahead
        5: 0.028,  # 2.8% for 5 months ahead
        6: 0.035,  # 3.5% for 6 months ahead
    }
    
    # Get growth rate for the specified month
    growth_rate = monthly_growth_rates.get(months_ahead, 0.005 * months_ahead)
    
    # Calculate future value
    future_value = current_value * (1 + growth_rate)
    
    return future_value

# Function to optimize buy price for a target profit
def optimize_buy_price(property_data, target_profit_pct, months_to_sell=0):
    """
    Optimize buy price to achieve target profit percentage.
    
    Parameters:
    -----------
    property_data : dict
        Dictionary containing property features
    target_profit_pct : float
        Target profit percentage
    months_to_sell : int
        Number of months before selling
        
    Returns:
    --------
    float
        Optimized buy price
    """
    # Get the predicted sale value
    if months_to_sell > 0:
        predicted_sale_value = forecast_future_value(property_data, months_to_sell)
    else:
        predicted_sale_value = predict_property_value(property_data, model, model_features)
    
    # Calculate optimal buy price to achieve target profit
    optimal_buy_price = predicted_sale_value / (1 + (target_profit_pct / 100))
    
    return optimal_buy_price, predicted_sale_value

# Function to create a property map
def create_property_map(property_data):
    """
    Create an interactive map showing the property location.
    
    Parameters:
    -----------
    property_data : dict
        Dictionary containing property features including lat/long
        
    Returns:
    --------
    folium.Map
        Interactive map object
    """
    # Create a map centered on the property
    lat = property_data.get('latitude', 34.0522)  # Default to LA if no coords
    lon = property_data.get('longitude', -118.2437)
    
    m = folium.Map(location=[lat, lon], zoom_start=15)
    
    # Add marker for the property
    address = property_data.get('address', 'Property Location')
    price = property_data.get('sale_price', 0)
    
    popup_text = f"""
    <b>Address:</b> {address}<br>
    <b>Price:</b> ${price:,.0f}<br>
    <b>Sqft:</b> {property_data.get('sqft', 0)}<br>
    <b>Beds/Baths:</b> {property_data.get('beds', 0)}/{property_data.get('baths', 0)}
    """
    
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)
    
    return m

# Function to create feature importance plot
def plot_feature_importance():
    """
    Create a bar chart of feature importance.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Feature importance plot
    """
    # Feature importance data (from your model analysis)
    features = [
        'ppsqft_log', 'relative_size', 'sqft', 'luxury_score', 'luxury_in_tier_2',
        'living_space_ratio', 'days_since_start', 'year_built', 'age_log', 'bed_bath_ratio'
    ]
    importance = [0.613063, 0.211105, 0.165541, 0.005618, 0.000705, 
                 0.000587, 0.000304, 0.000229, 0.000217, 0.000216]
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            colorbar=dict(title='Importance')
        )
    ))
    
    fig.update_layout(
        title='Top 10 Features by Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# Function to create price trend chart
def plot_price_trend(base_price, months=6):
    """
    Create a chart showing projected price trends.
    
    Parameters:
    -----------
    base_price : float
        Starting price for projections
    months : int
        Number of months to project
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Price trend chart
    """
    # Monthly growth rates (simplified for demonstration)
    monthly_growth = [0.005, 0.005, 0.006, 0.006, 0.007, 0.008]
    
    # Calculate projected values
    dates = [datetime.now() + timedelta(days=30*i) for i in range(months+1)]
    prices = [base_price]
    
    for i in range(months):
        next_price = prices[-1] * (1 + monthly_growth[min(i, len(monthly_growth)-1)])
        prices.append(next_price)
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines+markers',
        name='Projected Value',
        line=dict(color='rgba(50, 171, 96, 0.7)', width=2),
        marker=dict(size=8)
    ))
    
    # Add confidence interval (for demonstration)
    upper_bound = [p * 1.05 for p in prices]  # +5%
    lower_bound = [p * 0.95 for p in prices]  # -5%
    
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(50, 171, 96, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    fig.update_layout(
        title='Projected Property Value Over Time',
        xaxis_title='Date',
        yaxis_title='Projected Value ($)',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(tickformat='$,.0f'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Function to create profit optimization chart
def plot_profit_optimization(property_data, months_range=6):
    """
    Create a chart showing profit potential at different buy prices.
    
    Parameters:
    -----------
    property_data : dict
        Dictionary containing property features
    months_range : int
        Maximum number of months to consider
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Profit optimization chart
    """
    # Predict future values for different timeframes
    future_values = [forecast_future_value(property_data, m) for m in range(months_range+1)]
    
    # Calculate profitable buy prices for different profit targets
    profit_targets = [5, 10, 15, 20]
    buy_prices = {}
    
    for target in profit_targets:
        buy_prices[target] = [future_values[m] / (1 + (target/100)) for m in range(months_range+1)]
    
    # Create line chart
    fig = go.Figure()
    
    # Add future value line
    fig.add_trace(go.Scatter(
        x=list(range(months_range+1)),
        y=future_values,
        mode='lines+markers',
        name='Projected Sale Value',
        line=dict(color='rgba(50, 171, 96, 0.7)', width=3),
        marker=dict(size=8)
    ))
    
    # Add buy price lines for different profit targets
    colors = ['rgba(255, 127, 14, 0.7)', 'rgba(214, 39, 40, 0.7)', 
              'rgba(148, 103, 189, 0.7)', 'rgba(44, 160, 44, 0.7)']
    
    for i, target in enumerate(profit_targets):
        fig.add_trace(go.Scatter(
            x=list(range(months_range+1)),
            y=buy_prices[target],
            mode='lines',
            name=f'{target}% Profit Target',
            line=dict(color=colors[i], width=2, dash='dot'),
        ))
    
    fig.update_layout(
        title='Optimal Buy Price by Profit Target and Holding Period',
        xaxis_title='Months to Hold',
        yaxis_title='Price ($)',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(tickformat='$,.0f'),
        xaxis=dict(tickmode='array', tickvals=list(range(months_range+1))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">Real Estate Automated Valuation Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard provides advanced property valuation, profit optimization, and forecasting 
    capabilities using machine learning models trained on Southern California real estate data.
    """)
    
    # Sidebar for data upload and selection
    st.sidebar.title("Property Data")
    
    # Option to upload CSV or use default properties
    data_source = st.sidebar.radio(
        "Select Property Data Source",
        ["Use Default Properties", "Upload CSV"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Property CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                properties_df = pd.read_csv(uploaded_file)
                st.sidebar.success("CSV uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                properties_df = default_properties
        else:
            properties_df = default_properties
    else:
        properties_df = default_properties
    
    # Property selection dropdown
    selected_property_idx = st.sidebar.selectbox(
        "Select Property",
        options=range(len(properties_df)),
        format_func=lambda x: f"{properties_df.iloc[x]['address']} - ${properties_df.iloc[x]['sale_price']:,.0f}"
    )
    
    # Get selected property data
    selected_property = properties_df.iloc[selected_property_idx].to_dict()
    
    # Display property details
    st.sidebar.markdown("### Property Details")
    st.sidebar.markdown(f"**Address:** {selected_property['address']}")
    st.sidebar.markdown(f"**Price:** ${selected_property['sale_price']:,.0f}")
    st.sidebar.markdown(f"**Sq Ft:** {selected_property['sqft']}")
    st.sidebar.markdown(f"**Beds/Baths:** {selected_property['beds']}/{selected_property['baths']}")
    st.sidebar.markdown(f"**Year Built:** {selected_property['year_built']}")
    st.sidebar.markdown(f"**Lot Size:** {selected_property['lot_size']} sq ft")
    
    # Optimization parameters in sidebar
    st.sidebar.markdown("### Profit Optimization")
    target_profit = st.sidebar.slider("Target Profit (%)", min_value=1, max_value=30, value=10, step=1)
    holding_period = st.sidebar.selectbox("Holding Period (Months)", options=[0, 1, 2, 3, 6, 12])
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Profit Optimization", "Forecast", "Market Insights"])
    
    with tab1:
        # Property valuation section
        st.markdown('<div class="sub-header">Property Valuation</div>', unsafe_allow_html=True)
        
        # Create columns for metrics and map
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Display property image (simulated)
            image_placeholder = st.empty()
            try:
                # For demo, use a placeholder image based on property price
                if selected_property['sale_price'] > 1000000:
                    img_url = "https://images.unsplash.com/photo-1613490493576-7fde63acd811?auto=format&fit=crop&w=800&q=80"
                elif selected_property['sale_price'] > 750000:
                    img_url = "https://images.unsplash.com/photo-1592595896551-12b371d546d5?auto=format&fit=crop&w=800&q=80"
                else:
                    img_url = "https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=800&q=80"
                
                response = requests.get(img_url)
                image = Image.open(io.BytesIO(response.content))
                image_placeholder.image(image, use_column_width=True, caption=selected_property['address'])
            except:
                image_placeholder.info("Property image not available")
            
            # Predicted value
            predicted_value = predict_property_value(selected_property, model, model_features)
            actual_value = selected_property['sale_price']
            
            # Calculate valuation metrics
            valuation_diff = predicted_value - actual_value
            valuation_diff_pct = (valuation_diff / actual_value) * 100
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div class="metric-label">Current Market Value</div>
                    <div class="metric-value">${predicted_value:,.0f}</div>
                </div>
                <div>
                    <div class="metric-label">Listed Price</div>
                    <div class="metric-value">${actual_value:,.0f}</div>
                </div>
                <div>
                    <div class="metric-label">Difference</div>
                    <div class="metric-value {'profit-positive' if valuation_diff >= 0 else 'profit-negative'}">
                        {valuation_diff_pct:+.1f}% (${valuation_diff:,.0f})
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Property characteristics
            st.markdown("### Property Characteristics")
            
            # Create a 3-column layout for property features
            char_col1, char_col2, char_col3 = st.columns(3)
            
            with char_col1:
                st.metric("Square Footage", f"{selected_property['sqft']:,}")
                st.metric("Year Built", selected_property['year_built'])
                st.metric("Price/Sqft", f"${selected_property['sale_price']/selected_property['sqft']:.0f}")
            
            with char_col2:
                st.metric("Bedrooms", selected_property['beds'])
                st.metric("Lot Size", f"{selected_property['lot_size']:,} sqft")
                st.metric("Property Age", f"{datetime.now().year - selected_property['year_built']} years")
            
            with char_col3:
                st.metric("Bathrooms", selected_property['baths'])
                st.metric("Property Type", selected_property['property_type'])
                # Calculate bed/bath ratio
                st.metric("Bed/Bath Ratio", f"{selected_property['beds']/selected_property['baths']:.1f}")
        
        with col2:
            # Property location map
            st.markdown("### Location")
            property_map = create_property_map(selected_property)
            folium_static(property_map, width=400)
            
            # Neighborhood stats
            st.markdown("### Neighborhood Analysis")
            # For demo, use simulated stats
            neighborhood = "Sample Neighborhood"  # Would come from geocoding in a real app
            neighborhood_stats = {
                "Avg Sale Price": "$850,000",
                "Avg Price/Sqft": "$425",
                "Median Days on Market": "15",
                "Schools Rating": "8/10",
                "Walk Score": "75/100"
            }
            
            st.markdown(f"**{neighborhood}**")
            for stat, value in neighborhood_stats.items():
                st.markdown(f"**{stat}:** {value}")
        
        # Feature importance - under valuation tab
        st.markdown("### Valuation Drivers")
        st.markdown("The chart below shows the key features driving this property's valuation:")
        
        # Display feature importance
        feature_importance_plot = plot_feature_importance()
        st.plotly_chart(feature_importance_plot, use_container_width=True)
        
        # Add explanation of valuation drivers
        with st.expander("Explanation of Valuation Drivers"):
            st.markdown("""
            **Price per Square Foot (Log)**: The dominant driver of property value, accounting for over 60% of the valuation.
            
            **Relative Size**: How this property's size compares to neighborhood averages is the second most important factor (21%).
            
            **Square Footage**: The absolute size of the property contributes about 16.5% to the valuation.
            
            **Luxury Score**: A composite metric of premium features that indicates property quality beyond basic measurements.
            
            **Living Space Ratio**: The proportion of the lot that is living space, showing land utilization.
            """)
    
    with tab2:
        # Profit optimization section
        st.markdown('<div class="sub-header">Profit Optimization</div>', unsafe_allow_html=True)
        
        # Calculate optimized buy price
        optimized_price, projected_sale = optimize_buy_price(
            selected_property, target_profit, holding_period
        )
        
        # Display optimization results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between;">
            <div>
                <div class="metric-label">Optimal Buy Price</div>
                <div class="metric-value">${optimized_price:,.0f}</div>
                <div>For {target_profit}% profit target</div>
            </div>
            <div>
                <div class="metric-label">Projected Sale Price</div>
                <div class="metric-value">${projected_sale:,.0f}</div>
                <div>After {holding_period} months</div>
            </div>
            <div>
                <div class="metric-label">Market Assessment</div>
                <div class="metric-value {'profit-positive' if optimized_price >= selected_property['sale_price'] else 'profit-negative'}">
                    {('GOOD DEAL' if optimized_price >= selected_property['sale_price'] else 'OVERPRICED')}
                </div>
                <div>Listed at ${selected_property['sale_price']:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create columns for different visualizations
        opt_col1, opt_col2 = st.columns([3, 2])
        
        with opt_col1:
            # Profit optimization chart
            st.markdown("### Optimal Buy Price vs. Holding Period")
            profit_chart = plot_profit_optimization(selected_property)
            st.plotly_chart(profit_chart, use_container_width=True)
            
            st.markdown("""
            This chart shows the optimal buy price to achieve your target profit percentage
            based on different holding periods. The top line shows the projected sale value
            at each time point, while the dotted lines show buy prices needed for different
            profit targets.
            """)
        
        with opt_col2:
            # ROI Calculator
            st.markdown("### ROI Calculator")
            
            custom_buy_price = st.number_input(
            "Custom Purchase Price ($)",
            min_value=int(optimized_price * 0.5),
            max_value=int(optimized_price * 1.5),
            value=int(optimized_price),
            step=10000
            )
            
            # Additional costs
            renovation_costs = st.number_input("Renovation Costs ($)", min_value=0, value=0, step=5000)
            closing_costs = st.number_input("Closing Costs (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            closing_costs_dollar = custom_buy_price * (closing_costs / 100)
            
            # Calculate total investment
            total_investment = custom_buy_price + renovation_costs + closing_costs_dollar
            
            # Calculate ROI
            roi_dollar = projected_sale - total_investment
            roi_percent = (roi_dollar / total_investment) * 100
            
            # Display ROI metrics
            st.markdown(f"""
            **Total Investment:** ${total_investment:,.0f}  
            **Projected Sale:** ${projected_sale:,.0f}  
            **Profit:** ${roi_dollar:,.0f} ({roi_percent:.1f}%)
            """)
            
            # ROI gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=roi_percent,
                title={'text': "Return on Investment (%)"},
                gauge={
                    'axis': {'range': [-10, 30]},
                    'bar': {'color': "rgba(50, 171, 96, 0.7)"},
                    'steps': [
                        {'range': [-10, 0], 'color': "rgba(214, 39, 40, 0.6)"},
                        {'range': [0, 10], 'color': "rgba(255, 144, 14, 0.6)"},
                        {'range': [10, 20], 'color': "rgba(44, 160, 44, 0.6)"},
                        {'range': [20, 30], 'color': "rgba(22, 96, 167, 0.6)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target_profit
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Forecast section
        st.markdown('<div class="sub-header">Value Forecast</div>', unsafe_allow_html=True)
        
        forecast_col1, forecast_col2 = st.columns([2, 1])
        
        with forecast_col1:
            # Price trend chart
            trend_chart = plot_price_trend(predicted_value)
            st.plotly_chart(trend_chart, use_container_width=True)
        
        with forecast_col2:
            # Forecast table
            st.markdown("### Monthly Forecasts")
            
            forecast_data = []
            current_value = predicted_value
            
            for month in range(1, 7):
                future_val = forecast_future_value(selected_property, month)
                growth_pct = ((future_val / current_value) - 1) * 100
                profit_vs_purchase = ((future_val / selected_property['sale_price']) - 1) * 100
                
                forecast_data.append({
                    "Month": month,
                    "Value": future_val,
                    "Growth": growth_pct,
                    "vs Purchase": profit_vs_purchase
                })
            
            # Create DataFrame for table
            forecast_df = pd.DataFrame(forecast_data)
            
            # Format for display
            forecast_df["Value"] = forecast_df["Value"].apply(lambda x: f"${x:,.0f}")
            forecast_df["Growth"] = forecast_df["Growth"].apply(lambda x: f"{x:.1f}%")
            forecast_df["vs Purchase"] = forecast_df["vs Purchase"].apply(
                lambda x: f"<span class='{('profit-positive' if x >= 0 else 'profit-negative')}'>{x:.1f}%</span>"
            )
            
            # Display table with HTML formatting
            st.markdown(forecast_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Add disclaimer
            st.markdown("""
            <div style="font-size: 0.8rem; color: #666; margin-top: 1rem;">
            * Forecasts are based on historical market trends and current conditions. 
            Actual results may vary due to market fluctuations and other factors.
            </div>
            """, unsafe_allow_html=True)
        
        # Seasonality analysis
        st.markdown("### Seasonal Market Factors")
        
        # Create columns for seasonal charts
        season_col1, season_col2 = st.columns(2)
        
        with season_col1:
            # Seasonal price trends
            seasons_data = {
                'Season': ['Winter', 'Spring', 'Summer', 'Fall'],
                'Price Factor': [0.94, 1.05, 1.03, 0.98],
                'Days on Market': [45, 22, 30, 38]
            }
            
            seasons_df = pd.DataFrame(seasons_data)
            
            fig = px.bar(
                seasons_df,
                x='Season',
                y='Price Factor',
                color='Price Factor',
                color_continuous_scale='RdYlGn',
                text=seasons_df['Price Factor'].apply(lambda x: f"{(x-1)*100:+.1f}%")
            )
            
            fig.update_layout(
                title='Seasonal Price Impact',
                yaxis_title='Price Factor',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with season_col2:
            # Days on market by season
            fig = px.bar(
                seasons_df,
                x='Season',
                y='Days on Market',
                color='Days on Market',
                color_continuous_scale='RdYlGn_r',
                text=seasons_df['Days on Market'].apply(lambda x: f"{x} days")
            )
            
            fig.update_layout(
                title='Average Days on Market by Season',
                yaxis_title='Days',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Market insights section
        st.markdown('<div class="sub-header">Market Insights</div>', unsafe_allow_html=True)
        
        # Create columns for market visuals
        market_col1, market_col2 = st.columns([3, 2])
        
        with market_col1:
            # Model performance by price segment
            st.markdown("### Model Performance by Price Segment")
            
            # Data from your model analysis
            segments = [
                '($289,995, $671,011]',
                '($671,011, $1,056,102]',
                '($1,056,102, $1,441,193]', 
                '($1,441,193, $1,826,284]',
                '($1,826,284, $2,211,375]'
            ]
            
            errors = [1.246468, 0.595494, 2.204419, 20.258408, 27.423760]
            
            # Create bar chart
            fig = px.bar(
                x=segments,
                y=errors,
                color=errors,
                color_continuous_scale='RdYlGn_r',
                labels={'x': 'Price Segment', 'y': 'Mean Absolute Percentage Error (%)'},
                text=[f"{e:.2f}%" for e in errors]
            )
            
            fig.update_layout(
                title='Valuation Accuracy by Price Segment',
                xaxis_title='Price Range',
                yaxis_title='MAPE (%)',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Key Insight**: The AVM performs exceptionally well in the middle price segments, with errors below 2.5%. 
            High-end luxury properties (above $1.4M) are more challenging to automatically value due to their unique characteristics.
            """)
        
        with market_col2:
            # Price distribution in the market
            st.markdown("### Market Price Distribution")
            
            # Simplified price distribution for demo
            price_ranges = ['<$500K', '$500K-$750K', '$750K-$1M', '$1M-$1.5M', '$1.5M-$2M', '>$2M']
            distribution = [15, 25, 30, 20, 7, 3]  # Percentage of properties
            
            fig = px.pie(
                values=distribution,
                names=price_ranges,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            
            fig.update_layout(
                title='Property Price Distribution',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market trends
        st.markdown("### Market Trends")
        
        trends_col1, trends_col2, trends_col3 = st.columns(3)
        
        with trends_col1:
            # Months of inventory
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=3.2,
                number={'suffix': " months", "font": {"size": 32}},
                delta={'position': "bottom", 'reference': 4.5, 'relative': False},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={"text": "Months of Inventory"}
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("Market is favoring sellers with inventory below 4 months.")
        
        with trends_col2:
            # Median days on market
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=18,
                number={'suffix': " days", "font": {"size": 32}},
                delta={'position': "bottom", 'reference': 22, 'relative': False},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={"text": "Median Days on Market"}
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("Properties are selling faster than the 6-month average.")
        
        with trends_col3:
            # Median price trend
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=712000,
                number={'prefix': "$", 'valueformat': ",.0f", "font": {"size": 32}},
                delta={'position': "bottom", 'reference': 685000, 'relative': False, 'valueformat': ",.0f"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={"text": "Median Sale Price"}
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("Median prices up 3.9% from previous quarter.")
    
    # Footer section
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("""
    <p>This Automated Valuation Model (AVM) dashboard was developed for demonstration purposes. Results should be validated by a real estate professional before making investment decisions.</p>
    <p>© 2023 Real Estate Analytics Dashboard | Powered by Random Forest ML Model</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()