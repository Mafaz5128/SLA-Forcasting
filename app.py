import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go


# Function for the model
def Xgboost_model(df, sector, departure_date, forecast_period):
    # Step 1: Filter data for the selected sector
    df = df[df['Sector'] == sector]

    # Step 2: Group by 'Sale Date' and calculate average YLD USD and sum of PAX COUNT
    df = df.groupby("Sale Date", as_index=False).agg(
        Avg_YLD_USD=("YLD USD", "mean"),
        Sum_PAX=("PAX COUNT", "sum")
    )

    # Step 3: Calculate Days Before Departure
    df["Sale Date"] = pd.to_datetime(df["Sale Date"])
    departure_date = pd.to_datetime(departure_date)
    df["Days Before Departure"] = (departure_date - df["Sale Date"]).dt.days

    # Step 4: Filter rows before the forecast period
    forecast_date = pd.to_datetime(forecast_period)
    df = df[df["Sale Date"] <= forecast_date]

    # Step 5: Cumulative PAX COUNT
    df["Cumulative PAX COUNT"] = df["Sum_PAX"].cumsum()

    # Step 6: Feature Engineering
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

    # Split the dataset into training and testing sets
    train_data = df[df["Sale Date"] <= forecast_date]
    test_data = df[df["Sale Date"] > forecast_date]

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

    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Create Interactive Plot
    train_data["Predicted YLD USD"] = y_train_pred
    test_data["Predicted YLD USD"] = y_test_pred

    fig = go.Figure()

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

    return fig, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2


# Streamlit App
st.title("XGBoost Airline Yield Prediction")
st.write("This application allows you to forecast airline yield using XGBoost.")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file) 

    # Select parameters
    sector = st.selectbox("Select Sector", df["Sector"].unique())
    departure_date = st.date_input("Select Departure Date")
    forecast_period = st.date_input("Select Forecast Period End Date")

    if st.button("Run Forecast"):
        # Run the model
        fig, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2 = Xgboost_model(
            df, sector, departure_date, forecast_period
        )

        # Display results
        st.plotly_chart(fig)
        st.subheader("Model Performance")
        st.write(f"**Training RMSE:** {train_rmse:.2f}")
        st.write(f"**Testing RMSE:** {test_rmse:.2f}")
        st.write(f"**Training MAE:** {train_mae:.2f}")
        st.write(f"**Testing MAE:** {test_mae:.2f}")
        st.write(f"**Training R²:** {train_r2:.2f}")
        st.write(f"**Testing R²:** {test_r2:.2f}")
