# Tomato Price Predictor

## Overview

The **Tomato Price Predictor** is a Streamlit application designed to forecast onion prices based on historical data. It uses SARIMA (Seasonal AutoRegressive Integrated Moving Average) models to predict future prices and provides interactive visualizations for historical data.

## Features

- **Prediction**: Users can predict onion prices for a selected month and year.
- **Historical Data**: Users can view historical onion price data for different cities.
- **Interactive Interface**: Users can select the month, year, and city to get predictions and visualize past data.

## Requirements

- Python 3.7 or higher
- Streamlit
- pandas
- plotly
- statsmodels
- pickle

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/tomato_price_predictor.git
   cd tomato_price_predictor
Create and Activate a Virtual Environment

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the Required Packages

bash
Copy code
pip install -r requirements.txt
Prepare Your Data and Models

Place your train.csv file in the root directory.
Ensure that you have your model pickle files named as model_<City>.pkl in the root directory.
Usage
Run the Streamlit App

bash
Copy code
streamlit run app.py
Access the Application

Open your web browser and go to [http://localhost:8501](https://daam-dost-tomato-price-predictor.streamlit.app/) to interact with the application.

## Application Layout
- Title: Displays the application title.
- User Inputs:
   Month: Select the month for prediction.
   Year: Use the slider to select the year for prediction.
   City: Select the city for which you want to predict the price.
   Buttons:
      -- Predict: Click this to get the predicted onion price for the selected month and year.
      -- Show Past Data: Click this to visualize historical onion prices for the selected city.
## Styling
The application uses a gradient background with shades of orange, green, and blue, and ensures that text color is black for readability.

