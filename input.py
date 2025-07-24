import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set the parameters
store_id = 'Store_1'
start_date = '2022-01-01'
end_date = '2022-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='W-SUN')  # Weekly data

# Generate synthetic data
data = {
    'Date': date_range,
    'Store': [store_id] * len(date_range),
    'Weekly_Sales': np.random.randint(1000, 5000, size=len(date_range)),  # Random sales between 1000 and 5000
    'Unemployment': np.random.uniform(5.0, 10.0, size=len(date_range)),  # Random unemployment rate between 5% and 10%
    'Holiday_Flag': [random.choice([0, 1]) for _ in range(len(date_range))],  # Randomly assign holiday flag
    'CPI': np.random.uniform(150, 200, size=len(date_range))  # Random CPI values
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('store_sales_data.csv', index=False)

print("CSV file 'store_sales_data.csv' created successfully.")