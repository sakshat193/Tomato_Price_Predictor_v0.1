import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import numpy as np
# Streamlit app
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #ffe0b3, #b3e5fc); /* Orange to Blue gradient */
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50; /* Green */
        border-color: #4CAF50;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    .stSelectbox>div>div>div {
        background-color: #ff9933; /* Orange */
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50; /* Green */
    }
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
        color: #000000; /* Black text color */
    }
    .stMarkdown p, .stTitle, .stHeader, .stSubheader, .stText, .stSelectbox div, .stButton button {
        color: #000000; /* Black text color for all these elements */
    }
    .stText, .stMarkdown, .stHeader, .stSubheader {
        color: #000000; /* Ensure text in markdown, headers, and other elements is black */
    }
</style>
""", unsafe_allow_html=True)


# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Load the models
@st.cache_resource
def load_models():
    models = {}
    model_files = [
        "model_Varanasi.pkl", "model_Agra.pkl", "model_Ahmedabad.pkl", "model_Bathinda.pkl",
        "model_Ernakulam.pkl", "model_Gurgaon.pkl", "model_Jaipur.pkl", "model_Kanpur.pkl",
        "model_Lucknow.pkl", "model_Ludhiana.pkl", "model_Mumbai.pkl", "model_Nagpur.pkl",
        "model_Palakkad.pkl", "model_Shillong.pkl", "model_Shimla.pkl", "model_Siliguri.pkl",
        "model_Thrissur.pkl"
    ]
    for file in model_files:
        with open(file, 'rb') as f:
            models[file.split('_')[1].split('.')[0]] = pickle.load(f)
    return models

models = load_models()

# Streamlit app
st.title('Daam Dost - \nTomato Price Prediction and Historical Data')

# User inputs
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']
month = st.selectbox('Select Month', months)

# Get the min and max years from the data
min_year = df['Date'].dt.year.min()
max_year = datetime.now().year + 5  # Allow predictions up to 5 years in the future

year = st.slider('Select Year', min_value=min_year, max_value=max_year, value=datetime.now().year)

city = st.selectbox('Select City', list(models.keys()))

# Create two columns for buttons
col1, col2 = st.columns(2)

# Predict button
if col1.button('Predict'):
    try:
        # Prepare the date for prediction
        prediction_date = pd.Timestamp(year=year, month=months.index(month)+1, day=1)
        
        # Get the model for the selected city
        model = models[city]
        
        # Make prediction
        # We need to determine how many steps ahead our prediction date is
        last_date_in_data = df[df['Centre_Name'] == city]['Date'].max()
        steps = (prediction_date.to_period('M') - last_date_in_data.to_period('M')).n
        
        if steps > 0:
            forecast = model.get_forecast(steps=steps)
            predicted_price = forecast.predicted_mean[-1]  # Get the last predicted value
            
            # If the predicted price is a numpy array, get the first element
            if isinstance(predicted_price, np.ndarray):
                predicted_price = predicted_price[0]
            
            st.success(f'Predicted price of onions in {city} for {month} {year}: ₹{predicted_price:.2f} per kg')
        else:
            st.error("The prediction date is not in the future relative to the training data.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please check your input and try again.")

# Show Past Data button
if col2.button('Show Past Data'):
    # Filter data for the selected city
    city_data = df[df['Centre_Name'] == city]
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_data['Date'], y=city_data['Price'],
                             mode='lines', name='Price'))
    
    fig.update_layout(title=f'Historical Onion Prices in {city}',
                      xaxis_title='Date',
                      yaxis_title='Price (₹ per kg)')
    
    st.plotly_chart(fig)

st.info('Note: Predictions are based on historical data and may not account for recent market changes.')
