import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from itertools import product
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
st.header("Sector-wise Average Yield Prediction")

# File upload section
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["Sale Date"] = pd.to_datetime(df["Sale Date"])
    df["Flight Date"] = pd.to_datetime(df["Flight Date"])
    
    # Get unique sectors
    sectors = df['Sector'].unique()
    
    # Store results
    sector_avg_yield = []
    
    flight_dates = sorted(df['Flight Date'].unique())
    departure_date = st.selectbox('Select Departure Date', flight_dates)
    departure_date = pd.to_datetime(departure_date)
    
    last_sale_date = df['Sale Date'].max()
    forecast_window_start = max(last_sale_date, departure_date - pd.Timedelta(days=90))
    
    forecast_period_start = st.date_input(
        "Select Forecast Start Date",
        min_value=forecast_window_start,
        max_value=departure_date
    )
    
    forecast_period_end = st.date_input(
        "Select Forecast End Date",
        min_value=forecast_period_start,
        max_value=departure_date
    )
    
    if st.button("Generate Forecast"):
        forecast_results = []
        
        for selected_sector in sectors:
            df_filtered = df[df['Sector'] == selected_sector]
            
            df_grouped = df_filtered.groupby("Sale Date", as_index=False).agg(
                Avg_YLD_USD=("YLD USD", "mean")
            )
            
            df_forecast_data = df_grouped[df_grouped['Sale Date'] <= pd.Timestamp(forecast_period_start)]
            y_train = df_forecast_data["Avg_YLD_USD"]
            
            best_model = None
            best_rmse = float('inf')
            
            for trend, seasonal, seasonal_periods in product(['add', 'mul'], ['add', 'mul'], [7, 30]):
                try:
                    model = ExponentialSmoothing(y_train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
                    model_fit = model.fit()
                    rmse = mean_absolute_error(y_train, model_fit.fittedvalues)
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_fit
                except:
                    continue
            
            if best_model is not None:
                forecast_dates = pd.date_range(forecast_period_start, forecast_period_end, freq='D')
                y_pred_es = best_model.forecast(len(forecast_dates))
                
                forecast_results.append(pd.DataFrame({
                    "Sector": selected_sector,
                    "Sale Date": forecast_dates,
                    "Predicted Yield (Exp Smoothing)": y_pred_es
                }))
            else:
                st.warning(f"No suitable model found for sector: {selected_sector}. Skipping forecast.")
        
        if forecast_results:
            final_forecast_df = pd.concat(forecast_results)
            st.write("### Sector-wise Predicted Yield Forecast")
            st.dataframe(final_forecast_df)
            
            avg_yield_per_sector = final_forecast_df.groupby("Sector")["Predicted Yield (Exp Smoothing)"].mean().reset_index()
            avg_yield_per_sector.columns = ["Sector", "Average Predicted Yield (USD)"]
            st.write("### Sector-wise Average Predicted Yield Table")
            st.table(avg_yield_per_sector)
            
            # Download Button
            @st.cache_data
            def convert_df_to_excel(df):
                # Convert the DataFrame to Excel format
                return df.to_excel(index=False, engine='openpyxl')
            
            excel_file = convert_df_to_excel(avg_yield_per_sector)
            st.download_button(
                label="Download Sector-wise Average Predicted Yield Table",
                data=excel_file,
                file_name="sector_average_predicted_yield.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
