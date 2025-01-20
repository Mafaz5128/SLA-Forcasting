import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function for yield prediction and visualization
def yield_forecasting_app(df, sector, departure_date, split_date):
    st.title("Yield Forecasting Application")

    # Filter data based on sector and departure date
    st.write("### Data Overview")
    df = df[(df['Sector'] == sector) & (df['Departure Date'] == departure_date)]
    st.write(df.head())

    # Convert Sale Date to datetime
    df['Sale Date'] = pd.to_datetime(df['Sale Date'])

    # Feature Engineering
    df['Lag_1'] = df['Avg_YLD_USD'].shift(1)
    df['Lag_3'] = df['Avg_YLD_USD'].shift(3)
    df['MA_7'] = df['Avg_YLD_USD'].rolling(window=7).mean()
    df['EWMA_7'] = df['Avg_YLD_USD'].ewm(span=7, adjust=False).mean()
    df['EWMA_3'] = df['Avg_YLD_USD'].ewm(span=3, adjust=False).mean()

    # Drop rows with NaN values caused by lagging/rolling
    df.dropna(inplace=True)

    # Define features and target
    X = df[['Lag_1', 'Lag_3', 'MA_7', 'EWMA_3', 'EWMA_7', 'Sum_PAX', 'Days Before Departure', 'Cumulative PAX COUNT']]
    y = df['Avg_YLD_USD']

    # Split the dataset
    train_data = df[df['Sale Date'] <= split_date]
    test_data = df[df['Sale Date'] > split_date]

    X_train = train_data[X.columns]
    y_train = train_data['Avg_YLD_USD']
    X_test = test_data[X.columns]
    y_test = test_data['Avg_YLD_USD']

    # Initialize and train the model
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Performance Metrics
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    st.write("### Model Performance")
    st.write(f"Training RMSE: {train_rmse:.2f}, R2: {train_r2:.2f}")
    st.write(f"Testing RMSE: {test_rmse:.2f}, R2: {test_r2:.2f}")

    # Visualizations
    train_data['Predicted YLD USD'] = y_train_pred
    test_data['Predicted YLD USD'] = y_test_pred

    st.write("### Actual vs Predicted YLD USD")
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['Sale Date'], train_data['Avg_YLD_USD'], label='Actual YLD USD (Train)', color='blue', alpha=0.6)
    plt.plot(train_data['Sale Date'], train_data['Predicted YLD USD'], label='Predicted YLD USD (Train)', color='green', linestyle='--', alpha=0.8)
    plt.plot(test_data['Sale Date'], test_data['Avg_YLD_USD'], label='Actual YLD USD (Test)', color='orange', alpha=0.6)
    plt.plot(test_data['Sale Date'], test_data['Predicted YLD USD'], label='Predicted YLD USD (Test)', color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Sale Date')
    plt.ylabel('YLD USD')
    plt.title('Actual vs Predicted YLD USD over Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Residual Plots
    st.write("### Residuals")
    train_data['Residuals'] = y_train - y_train_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(train_data['Sale Date'], train_data['Residuals'], color='blue', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residual Plot for Training Data')
    plt.xlabel('Sale Date')
    plt.ylabel('Residuals')
    plt.grid(True)
    st.pyplot(plt)

    test_data['Residuals'] = y_test - y_test_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(test_data['Sale Date'], test_data['Residuals'], color='red', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residual Plot for Testing Data')
    plt.xlabel('Sale Date')
    plt.ylabel('Residuals')
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app setup
st.sidebar.title("Inputs")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xls", "xlsx"])

if uploaded_file:
    # Load the dataset from the uploaded file
    excel_data = pd.ExcelFile(uploaded_file)

    # Let the user select the sheet
    sheet_name = st.sidebar.selectbox("Select Sheet", excel_data.sheet_names)
    df = excel_data.parse(sheet_name)

    # Let the user select sector from the unique sectors in the dataset
    sector = st.sidebar.selectbox("Select Sector", df['Sector'].unique())

    # Let the user select dates
    departure_date = st.sidebar.date_input("Select Departure Date", min_value=df['Departure Date'].min(), max_value=df['Departure Date'].max())
    split_date = st.sidebar.date_input("Select Train-Test Split Date", min_value=df['Sale Date'].min(), max_value=df['Sale Date'].max())

    # Call the forecasting function
    yield_forecasting_app(df, sector, departure_date, split_date)
