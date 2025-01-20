import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

def Xgboost_model(df, sector, departure_date, forecast_period_start, forecast_period_end):
    # Convert 'Sale Date' to datetime if it's not already in datetime format
    df["Sale Date"] = pd.to_datetime(df["Sale Date"])
    
    # Ensure forecast_period_end is a datetime object
    forecast_period_end = pd.to_datetime(forecast_period_end)

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
    X = df[[
        "Lag_1", "Lag_3", "MA_7", "EWMA_3", "EWMA_7", 
        "Sum_PAX", "Days Before Departure", "Cumulative PAX COUNT"
    ]]
    y = df["Avg_YLD_USD"]
    forecast_period_start = pd.to_datetime(forecast_period_start)

    # Split the dataset into training and testing sets
    train_data = df[df["Sale Date"] <= forecast_period_start]
    test_data = df[df["Sale Date"] > forecast_period_start]

    X_train = train_data[X.columns]
    y_train = train_data["Avg_YLD_USD"]

    X_test = test_data[X.columns]
    y_test = test_data["Avg_YLD_USD"]

    # Initialize and train the XGBoost model
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Add predictions to train and test dataframes
    train_data["Predicted YLD USD"] = y_train_pred
    test_data["Predicted YLD USD"] = y_test_pred

    # Create traces for the interactive plot
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(
        x=train_data["Sale Date"],
        y=train_data["Avg_YLD_USD"],
        mode="lines",
        name="Actual (Train)",
        line=dict(color="blue"),
    ))

    fig.add_trace(go.Scatter(
        x=train_data["Sale Date"],
        y=train_data["Predicted YLD USD"],
        mode="lines",
        name="Predicted (Train)",
        line=dict(color="green", dash="dot"),
    ))

    # Add testing data
    fig.add_trace(go.Scatter(
        x=test_data["Sale Date"],
        y=test_data["Avg_YLD_USD"],
        mode="lines",
        name="Actual (Test)",
        line=dict(color="orange"),
    ))

    fig.add_trace(go.Scatter(
        x=test_data["Sale Date"],
        y=test_data["Predicted YLD USD"],
        mode="lines",
        name="Predicted (Test)",
        line=dict(color="red", dash="dot"),
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


# Streamlit user interface
st.title("XGBoost Model for Forecasting YLD USD")

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

    # Run the model
    if st.sidebar.button("Run Model"):
        Xgboost_model(df, sector, departure_date, forecast_period_start, forecast_period_end)
