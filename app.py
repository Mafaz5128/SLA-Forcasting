import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

# Streamlit page configuration
st.set_page_config(
    page_title="Station Revenue Analysis Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

def combined_model(df, sector, departure_date, forecast_period_start, forecast_period_end):
    # Convert 'Sale Date' to datetime if it's not already in datetime format
    df["Sale Date"] = pd.to_datetime(df["Sale Date"])
    forecast_period_end = pd.to_datetime(forecast_period_end)
    forecast_period_start = pd.to_datetime(forecast_period_start)

    # Filter data for the selected sector
    df = df[df['Sector'] == sector]

    # Group by 'Sale Date' and calculate average YLD USD and sum of PAX COUNT
    df = df.groupby("Sale Date", as_index=False).agg(
        Avg_YLD_USD=("YLD USD", "mean"),
        Sum_PAX=("PAX COUNT", "sum")
    )

    # Calculate Days Before Departure
    departure_date = pd.to_datetime(departure_date)
    df["Days Before Departure"] = (departure_date - df["Sale Date"]).dt.days
    df = df[df["Sale Date"] <= forecast_period_end]

    # Cumulative PAX COUNT
    df["Cumulative PAX COUNT"] = df["Sum_PAX"].cumsum()

    # Feature Engineering
    df["Lag_1"] = df["Avg_YLD_USD"].shift(1)
    df["Lag_3"] = df["Avg_YLD_USD"].shift(3)
    df["MA_7"] = df["Avg_YLD_USD"].rolling(window=7).mean()
    df["EWMA_3"] = df["Avg_YLD_USD"].ewm(span=3, adjust=False).mean()
    df["EWMA_7"] = df["Avg_YLD_USD"].ewm(span=7, adjust=False).mean()

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Define features and target
    X = df[["Lag_1", "Lag_3", "MA_7", "EWMA_3", "EWMA_7", 
             "Sum_PAX", "Days Before Departure", "Cumulative PAX COUNT"]]
    y = df["Avg_YLD_USD"]

    # Split the dataset into training and testing sets
    train_data = df[df["Sale Date"] <= forecast_period_start]
    test_data = df[df["Sale Date"] > forecast_period_start]

    X_train = train_data[X.columns]
    y_train = train_data["Avg_YLD_USD"]

    X_test = test_data[X.columns]
    y_test = test_data["Avg_YLD_USD"]

    # Random Forest Model
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)

    # XGBoost Model
    xgb_model = XGBRegressor(objective="reg:squarederror")
    xgb_model.fit(X_train, y_train)
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)

    # Add predictions to dataframes
    train_data["RF_Predicted_YLD_USD"] = rf_train_pred
    test_data["RF_Predicted_YLD_USD"] = rf_test_pred

    train_data["XGB_Predicted_YLD_USD"] = xgb_train_pred
    test_data["XGB_Predicted_YLD_USD"] = xgb_test_pred

    return train_data, test_data

# Streamlit user interface
st.title("Revenue Analysis: Combined XGBoost and Random Forest Models")

# File upload for CSV or Excel
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        # If the file is Excel, load it and let the user select a sheet
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        sheet_name = st.selectbox("Select Sheet", sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet_name)

    # Inputs
    sector = st.sidebar.selectbox("Select Sector", df['Sector'].unique())
    departure_date = st.sidebar.date_input("Departure Date")
    forecast_period_start = st.sidebar.date_input("Forecast Period Start")
    forecast_period_end = st.sidebar.date_input("Forecast Period End")

    if st.sidebar.button("Run Combined Models"):
        train_data, test_data = combined_model(df, sector, departure_date, forecast_period_start, forecast_period_end)

        # Create traces for the interactive plot
        fig = go.Figure()

        # Add training data (Actual)
        fig.add_trace(go.Scatter(
            x=train_data["Sale Date"],
            y=train_data["Avg_YLD_USD"],
            mode="lines",
            name="Actual (Train)",
            line=dict(color="blue"),
        ))

        # Add testing data (Actual)
        fig.add_trace(go.Scatter(
            x=test_data["Sale Date"],
            y=test_data["Avg_YLD_USD"],
            mode="lines",
            name="Actual (Test)",
            line=dict(color="orange"),
        ))

        # Add Random Forest Predictions (Test)
        fig.add_trace(go.Scatter(
            x=test_data["Sale Date"],
            y=test_data["RF_Predicted_YLD_USD"],
            mode="lines",
            name="RF Predicted (Test)",
            line=dict(color="green", dash="dot"),
        ))

        # Add XGBoost Predictions (Test)
        fig.add_trace(go.Scatter(
            x=test_data["Sale Date"],
            y=test_data["XGB_Predicted_YLD_USD"],
            mode="lines",
            name="XGB Predicted (Test)",
            line=dict(color="red", dash="dash"),
        ))

        # Update layout
        fig.update_layout(
            title="Actual vs Predicted YLD USD Over Time",
            xaxis_title="Sale Date",
            yaxis_title="YLD USD",
            legend_title="Legend",
            template="plotly_white",
            hovermode="x unified",
            width=1000,
            height=600
        )

        st.plotly_chart(fig)

        # Display data tables
        st.subheader("Train Data: Actual vs Predicted")
        st.dataframe(train_data[["Sale Date", "Avg_YLD_USD", "RF_Predicted_YLD_USD", "XGB_Predicted_YLD_USD"]])

        st.subheader("Test Data: Actual vs Predicted")
        st.dataframe(test_data[["Sale Date", "Avg_YLD_USD", "RF_Predicted_YLD_USD", "XGB_Predicted_YLD_USD"]])
