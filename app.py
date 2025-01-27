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
    flight_date = st.date_input("Select a Flight Date:", min_value=pd.to_datetime(filtered_data['Flight Date']).min(), max_value=pd.to_datetime(filtered_data['Flight Date']).max())
    filtered_data = filtered_data[filtered_data['Flight Date'] == str(flight_date)]

    # Filter by sale date
    sale_date = st.date_input("Select Sale Date Range:",
                              min_value=pd.to_datetime(filtered_data['Sale Date']).min(),
                              max_value=pd.to_datetime(filtered_data['Sale Date']).max(),
                              value=(pd.to_datetime(filtered_data['Sale Date']).min(), pd.to_datetime(filtered_data['Sale Date']).max()))
    filtered_data = filtered_data[(pd.to_datetime(filtered_data['Sale Date']) >= sale_date[0]) &
                                  (pd.to_datetime(filtered_data['Sale Date']) <= sale_date[1])]

    # Group data
    grouped_data = filtered_data.groupby("Sale Date", as_index=False).agg(
        Avg_YLD_USD=("YLD USD", "mean"),
        Sum_PAX=("PAX COUNT", "sum")
    )

    grouped_data["Sale Date"] = pd.to_datetime(grouped_data["Sale Date"])
    departure_date = pd.to_datetime(flight_date)
    grouped_data["Days Before Departure"] = (departure_date - grouped_data["Sale Date"]).dt.days

    st.write("### Processed Data:")
    st.dataframe(grouped_data)

    # Forecast period
    forecast_period = st.slider("Select Forecast Period (in days):", min_value=1, max_value=30, value=10)

    # Exponential Smoothing
    model = ExponentialSmoothing(grouped_data['Avg_YLD_USD'], trend='add', seasonal='add', seasonal_periods=7)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=forecast_period)
    forecast_reversed = forecast[::-1]

    # Plot
    st.write("### Yield Forecasting")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training data
    ax.plot(grouped_data['Days Before Departure'], grouped_data['Avg_YLD_USD'], marker='o', linestyle='-', label='Training Data')
    
    # Forecast
    forecast_index = np.arange(grouped_data['Days Before Departure'].max(), grouped_data['Days Before Departure'].max() - forecast_period, -1)
    ax.plot(forecast_index, forecast_reversed, marker='o', linestyle='-', color='red', label='Forecast')

    ax.set_xlabel('Days Before Departure')
    ax.set_ylabel('YLD USD')
    ax.invert_xaxis()  # Reverse the X-axis
    ax.legend()
    ax.set_title('Training Data and Forecast')
    st.pyplot(fig)

    # Comparison table
    st.write("### Forecast vs Actual Data")
    forecasted_vs_actual = pd.DataFrame({
        'Forecasted': forecast_reversed[:len(grouped_data)],
        'Actual': grouped_data['Avg_YLD_USD'].values[:len(forecast_reversed)]
    })
    st.dataframe(forecasted_vs_actual)
