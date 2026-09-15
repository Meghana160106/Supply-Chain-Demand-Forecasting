import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Supply Chain Demand Forecasting")
st.markdown(
    "A machine learning application for analyzing historical demand "
    "and forecasting future supply requirements."
)

uploaded = st.file_uploader(
    "Upload demand data (CSV or ZIP)",
    type=["csv", "zip"]
)

def create_sample_data():
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    demand = (
        100
        + np.linspace(0, 30, 180)
        + 15 * np.sin(np.arange(180) / 7)
        + np.random.normal(0, 8, 180)
    )

    return pd.DataFrame({
        "Date": dates,
        "Demand": np.maximum(demand, 10).round(0)
    })

if uploaded:

    if uploaded.name.lower().endswith(".zip"):
        files = []

        with zipfile.ZipFile(uploaded, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    files.append(name)

            if not files:
                st.error("No CSV file found inside the ZIP file.")
                st.stop()

            selected_file = st.selectbox(
                "Select CSV dataset",
                files
            )

            data = pd.read_csv(
                io.BytesIO(z.read(selected_file))
            )

    else:
        data = pd.read_csv(uploaded)

else:
    st.info("No dataset uploaded. A sample supply-chain dataset is being used.")
    data = create_sample_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

date_column = None
demand_column = None

for col in data.columns:
    name = col.lower().strip()

    if date_column is None and any(
        x in name for x in ["date", "day", "time", "timestamp"]
    ):
        date_column = col

    if demand_column is None and any(
        x in name for x in [
            "demand",
            "sales",
            "quantity",
            "units",
            "orders"
        ]
    ):
        demand_column = col

if date_column is None:
    date_column = st.selectbox(
        "Select date column",
        data.columns
    )

if demand_column is None:
    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        st.error("Dataset must contain a numerical demand column.")
        st.stop()

    demand_column = st.selectbox(
        "Select demand column",
        numeric_columns
    )

df = data[[date_column, demand_column]].copy()

df.columns = ["Date", "Demand"]

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Demand"] = pd.to_numeric(
    df["Demand"],
    errors="coerce"
)

df = df.dropna()

df = df.groupby("Date", as_index=False)["Demand"].sum()

df = df.sort_values("Date")

if len(df) < 15:
    st.error("At least 15 valid observations are required.")
    st.stop()

df["lag_1"] = df["Demand"].shift(1)
df["lag_7"] = df["Demand"].shift(7)
df["rolling_7"] = df["Demand"].shift(1).rolling(7).mean()
df["day_of_week"] = df["Date"].dt.dayofweek

model_data = df.dropna().copy()

features = [
    "lag_1",
    "lag_7",
    "rolling_7",
    "day_of_week"
]

X = model_data[features]
y = model_data["Demand"]

split = int(len(model_data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

st.subheader("📈 Demand Forecast")

forecast_df = pd.DataFrame({
    "Date": model_data["Date"].iloc[split:],
    "Actual Demand": y_test.values,
    "Forecast Demand": predictions.round(1)
})

st.line_chart(
    forecast_df.set_index("Date")
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Average Demand",
    f"{df['Demand'].mean():.1f}"
)

c2.metric(
    "MAE",
    f"{mae:.2f}"
)

c3.metric(
    "RMSE",
    f"{rmse:.2f}"
)

st.subheader("🔮 Future Demand Forecast")

future_days = st.slider(
    "Forecast horizon",
    7,
    30,
    14
)

history = df["Demand"].tolist()
future_values = []

for i in range(future_days):

    lag_1 = history[-1]

    lag_7 = history[-7]

    rolling_7 = np.mean(history[-7:])

    next_date = df["Date"].max() + pd.Timedelta(days=i + 1)

    day_of_week = next_date.dayofweek

    row = pd.DataFrame([{
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_7": rolling_7,
        "day_of_week": day_of_week
    }])

    prediction = model.predict(row)[0]

    future_values.append(prediction)

    history.append(prediction)

future_dates = pd.date_range(
    df["Date"].max() + pd.Timedelta(days=1),
    periods=future_days
)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast Demand": np.round(future_values, 1)
})

st.dataframe(
    future_df,
    use_container_width=True
)

st.subheader("💡 Supply Chain Insights")

average_future = np.mean(future_values)

if average_future > df["Demand"].mean() * 1.15:
    st.warning(
        "Forecast demand is significantly higher than historical demand. "
        "Consider increasing inventory and procurement."
    )
elif average_future < df["Demand"].mean() * 0.85:
    st.info(
        "Forecast demand is lower than historical demand. "
        "Inventory levels can be reviewed to reduce excess stock."
    )
else:
    st.success(
        "Forecast demand remains close to historical levels. "
        "Current supply planning can be maintained with regular monitoring."
    )

st.subheader("📥 Download Forecast")

csv = future_df.to_csv(index=False)

st.download_button(
    "Download Forecast CSV",
    csv,
    "demand_forecast.csv",
    "text/csv"
)

st.divider()

st.caption(
    "Supply Chain Demand Forecasting | Machine Learning Research Prototype"
)