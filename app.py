import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px

# Streamlit page configuration
st.set_page_config(
    page_title="Average YLD Forecasting Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

def combined_model(df, sector, departure_date, forecast_period_start, forecast_period_end):
    # Convert 'Sale Date' to datetime
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
    X = df[["Lag_1", "Lag_3", "MA_7", "EWMA_3", "EWMA_7", 
            "Sum_PAX", "Days Before Departure", "Cumulative PAX COUNT"]]
    y = df["Avg_YLD_USD"]

    forecast_period_start = pd.to_datetime(forecast_period_start)

    # Split the dataset into training and testing sets
    train_data = df[df["Sale Date"] <= forecast_period_start]
    test_data = df[df["Sale Date"] > forecast_period_start]

    X_train = train_data[X.columns]
    y_train = train_data["Avg_YLD_USD"]

    X_test = test_data[X.columns]
    y_test = test_data["Avg_YLD_USD"]

    # Initialize and train the RandomForest model
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)

    # Predict with Random Forest
    y_test_pred_rf = rf_model.predict(X_test)

    # Initialize and train the XGBoost model
    xgb_model = XGBRegressor(objective="reg:squarederror")
    xgb_model.fit(X_train, y_train)

    # Predict with XGBoost
    y_test_pred_xgb = xgb_model.predict(X_test)

    # Add predictions to test dataframe
    test_data["Predicted YLD USD (RF)"] = y_test_pred_rf
    test_data["Predicted YLD USD (XGB)"] = y_test_pred_xgb

    return train_data, test_data, df

# Streamlit user interface
st.title("Average YLD Prediction")

# Load the dataset
file_path = "Daily Yield_Nov24_12M&6M.xlsx"
df = pd.read_excel(file_path, sheet ='6M')

# Sidebar for user inputs
sector = st.sidebar.selectbox("Select Sector", df['Sector'].unique())
forecast_period_start = st.sidebar.date_input("Forecast Period: Start")
forecast_period_end = st.sidebar.date_input("Forecast Period: End")
departure_date = st.sidebar.date_input("Departure Date")

# Run the combined model and display results
if st.sidebar.button("Forecast"):
    train_data, test_data, df = combined_model(
        df, sector, departure_date, forecast_period_start, forecast_period_end
    )

    # Create tabs for the chart and the tables
    tab1, tab2 = st.tabs(["Chart", "Table"])

    with tab1:
        # Create traces for the interactive plot
        fig = go.Figure()

        # Add training data (Actual values)
        fig.add_trace(go.Scatter(
            x=train_data["Sale Date"],
            y=train_data["Avg_YLD_USD"],
            mode="lines",
            name="Actual (Train)",
            line=dict(color="blue"),
        ))

        # Add testing data (Actual values)
        fig.add_trace(go.Scatter(
            x=test_data["Sale Date"],
            y=test_data["Avg_YLD_USD"],
            mode="lines",
            name="Actual (Test)",
            line=dict(color="orange"),
        ))

        # Add predictions for Random Forest
        fig.add_trace(go.Scatter(
            x=test_data["Sale Date"],
            y=test_data["Predicted YLD USD (RF)"],
            mode="lines",
            name="Predicted (RF)",
            line=dict(color="green", dash="dot"),
        ))

        # Add predictions for XGBoost
        fig.add_trace(go.Scatter(
            x=test_data["Sale Date"],
            y=test_data["Predicted YLD USD (XGB)"],
            mode="lines",
            name="Predicted (XGB)",
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

    with tab2:
        st.subheader("Test Data: Actual vs Predicted")
        st.dataframe(test_data[["Sale Date", "Avg_YLD_USD", "Predicted YLD USD (RF)", "Predicted YLD USD (XGB)"]])

    # Filter for PAX graph dynamically
    df["Sale Date"] = pd.to_datetime(df["Sale Date"])
    forecast_period_end = pd.to_datetime(forecast_period_end)
    df = df[df['Sector'] == sector]
    df = df.groupby("Sale Date", as_index=False).agg(
        Avg_YLD_USD=("YLD USD", "mean"),
        Sum_PAX=("PAX COUNT", "sum")
    )
    departure_date = pd.to_datetime(departure_date)
    df["Days Before Departure"] = (departure_date - df["Sale Date"]).dt.days
    df = df[df["Sale Date"] <= forecast_period_end]
    df["Cumulative PAX COUNT"] = df["Sum_PAX"].cumsum()

    fig2 = px.line(
        df,
        x="Sale Date",
        y="Cumulative PAX COUNT",
        title=f"Cumulative PAX COUNT Before {forecast_period_start} for Sector: {sector}",
        labels={"Sale Date": "Sale Date", "Cumulative PAX COUNT": "Cumulative PAX COUNT"},
        markers=True
    )

    # Update layout for better visuals
    fig2.update_layout(
        xaxis_title="Sale Date",
        yaxis_title="Cumulative PAX COUNT",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig2)
