import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate a date range for the past 2 years (adjust the date range as needed)
dates = pd.date_range(start="2022-01-01", end=datetime.today(), freq='B')  # 'B' stands for business days

# Generate random stock prices (you can replace this with actual data)
np.random.seed(42)
prices = [round(random.uniform(100, 500), 2) for _ in range(len(dates))]  # Simulated stock prices between 100 and 500

# Create a DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Amazon_Price': prices
})

# Save the data to a CSV file
data.to_csv('Stock Market Dataset.csv', index=False)

print("Sample stock data has been saved to 'Stock Market Dataset.csv'")
