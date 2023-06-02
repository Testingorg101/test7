import pandas as pd
from prophet import Prophet

# Load the data
data = pd.read_csv('data.csv')
data['ds'] = pd.to_datetime(data['ds'])

# Create a Prophet model
model = Prophet()

# Fit the model with the data
model.fit(data)

# Generate future dates for forecasting
future_dates = model.make_future_dataframe(periods=30)  # Forecast for 30 days

# Make predictions
forecast = model.predict(future_dates)

# Visualize the forecast
model.plot(forecast)
model.plot_components(forecast)
