import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Streamlit app
st.title('Yield Forecasting with Exponential Smoothing')

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file:
    # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Uploaded Dataset:")
    st.dataframe(df.head())

    # Filter by sector
    sector = st.selectbox("Select a Sector:", df['Sector'].unique())
    filtered_data = df[df['Sector'] == sector]

    # Filter by flight date
    flight_date = st.date_input("Select a Flight Date:", 
                                min_value=pd.to_datetime(filtered_data['Flight Date']).min(), 
                                max_value=pd.to_datetime(filtered_data['Flight Date']).max())
    filtered_data = filtered_data[filtered_data['Flight Date'] == str(flight_date)]

    # Select forecast period based on sale dates
    min_sale_date = pd.to_datetime(filtered_data['Sale Date']).min()
    max_sale_date = pd.to_datetime(filtered_data['Sale Date']).max()

    forecast_start = st.date_input("Select Forecast Start Date:", 
                                   min_value=min_sale_date, 
                                   max_value=max_sale_date, 
                                   value=min_sale_date)
    forecast_end = st.date_input("Select Forecast End Date:", 
                                 min_value=forecast_start, 
                                 max_value=max_sale_date, 
                                 value=max_sale_date)

    # Filter data based on forecast period
    filtered_data['Sale Date'] = pd.to_datetime(filtered_data['Sale Date'])
    forecast_data = filtered_data[(filtered_data['Sale Date'] >= forecast_start) & 
                                  (filtered_data['Sale Date'] <= forecast_end)]

    # Group data
    grouped_data = forecast_data.groupby("Sale Date", as_index=False).agg(
        Avg_YLD_USD=("YLD USD", "mean"),
        Sum_PAX=("PAX COUNT", "sum")
    )

    st.write("### Processed Data:")
    st.dataframe(grouped_data)

    # Exponential Smoothing
    model = ExponentialSmoothing(grouped_data['Avg_YLD_USD'], trend='add', seasonal='add', seasonal_periods=7)
    fitted_model = model.fit()

    # Forecast period
    forecast_period = (pd.to_datetime(forecast_end) - pd.to_datetime(forecast_start)).days
    forecast = fitted_model.forecast(steps=forecast_period)

    # Plot
    st.write("### Yield Forecasting")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training data
    ax.plot(grouped_data['Sale Date'], grouped_data['Avg_YLD_USD'], marker='o', linestyle='-', label='Training Data')
    
    # Forecast
    forecast_dates = pd.date_range(start=forecast_start, periods=forecast_period, freq='D')
    ax.plot(forecast_dates, forecast, marker='o', linestyle='-', color='red', label='Forecast')

    ax.set_xlabel('Sale Date')
    ax.set_ylabel('YLD USD')
    ax.legend()
    ax.set_title('Training Data and Forecast')
    st.pyplot(fig)

    # Comparison table
    st.write("### Forecast vs Actual Data")
    forecasted_vs_actual = pd.DataFrame({
        'Sale Date': forecast_dates[:len(grouped_data)],
        'Forecasted': forecast[:len(grouped_data)],
        'Actual': grouped_data['Avg_YLD_USD'].values[:len(forecast)]
    })
    st.dataframe(forecasted_vs_actual)
