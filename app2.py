import streamlit as st
import pandas as pd 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
def app():
   st.markdown("""
        <style>
        .animated-header {
            animation: fadeIn 4s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)
    # Use the animated class in your header
   st.markdown('<h1 class="animated-header">U.I Sales Prediction..!</h1>', unsafe_allow_html=True)
   st.info("This  module is under testing!..")

   warnings.filterwarnings("ignore", category=FutureWarning)

   # Title of the app
   st.title("Sales Analysis")

   # File uploader for the CSV file
   uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

   #check file whether it has desired columns.
   st.info("Required Columns: 'Date', 'Store', 'Weekly_Sales','Unemployment', 'Holiday_Flag','CPI'")
   required_columns = ['Date', 'Store', 'Weekly_Sales','Unemployment', 'Holiday_Flag','CPI']

   #MAIN FUNCTIONS
   if uploaded_file is not None:
      try:
         # Read the CSV file
         walmart_data = pd.read_csv(uploaded_file)
         missing_columns = [col for col in required_columns if col not in walmart_data.columns]

         # Convert 'Date' to datetime
         walmart_data['Date'] = pd.to_datetime(walmart_data['Date'], format="%d-%m-%Y", errors='coerce')

         # Check for any NaT values in the Date column
         if walmart_data['Date'].isnull().any():
               st.warning("Some dates could not be converted and will be set to NaT (Not a Time).")

         # Clean and Preprocess the Data
         # Strip whitespace from the column names
         walmart_data.columns = walmart_data.columns.str.strip()
         if 'Store' in walmart_data.columns:
               # Calculate the total number of unique stores
               unique_stores = walmart_data['Store'].unique()
               num_stores = len(unique_stores)
               
               # Display the total number of stores
               st.write(f"Total number of unique stores present in the uploaded file: {num_stores}")
         else:
               st.warning("The uploaded file does not contain a 'store' column.")
         # Check for required columns
         missing_columns = [col for col in required_columns if col not in walmart_data.columns]
         if missing_columns:
               st.warning(f"The following required columns are missing from the uploaded file: {', '.join(missing_columns)}")
         else:
               # Proceed with further analysis if all required columns are present
               st.success("All required columns are present. Proceeding with analysis...")

               # Display the dataframe
               with st.expander('Data Preview'):
                  st.write(walmart_data.head())
               # Display descriptive statistics
               with st.expander('Descriptive Statistics'):
                  description = walmart_data.describe(include='all')
                  st.write(description)

               left_column, right_column = st.columns(2)
               # weekly sales
               #  ly_chart(fig)Create the bar plot
               fig = px.bar(data_frame=walmart_data, x='Store', y='Weekly_Sales', title='<b>Weekly Sales by Store</b>')
        
               # Display the plot in Streamlit
    
               left_column.plotly_chart(fig,use_container_width = True)
               
               #YEARLY SALES
               # Extract Year and Month from Date
               walmart_data['Year'] = walmart_data['Date'].dt.year
               walmart_data['Month'] = walmart_data['Date'].dt.strftime('%Y-%m')  # Format as 'YYYY-MM'
               # Group by Year and Month to calculate total Weekly Sales
               year_month_sales = walmart_data.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
               # Create the bar plot
            
               fig1 = px.bar(data_frame=year_month_sales, x='Month', y='Weekly_Sales', color='Year', title='<b>Monthly Weekly Sales by Year</b>')

               # Display the plot in Streamlit
               
               right_column.plotly_chart(fig1,use_container_width = True)

               #heatmap
               if 'Fuel_Price' in walmart_data.columns and 'CPI' in walmart_data.columns:
                  # Convert 'Fuel_Price' and 'CPI' to float
                  walmart_data['Fuel_Price'] = walmart_data['Fuel_Price'].replace({'\$': '', ' ': ''}, regex=True).astype(float)
                  walmart_data['CPI'] = walmart_data['CPI'].replace({'\$': '', ' ': ''}, regex=True).astype(float)  # If applicable

                  # Select numeric data for correlation
                  numeric_data = walmart_data.select_dtypes(include=['float64', 'int64'])

                  # Create a heatmap
                  plt.figure(figsize=(10, 8))
                  sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f', linewidths=.8, cmap='coolwarm')
                  plt.title('Correlation Heatmap of Walmart Data')

                  # Display the heatmap in Streamlit
                  with st.expander("Correlation Heatmap"):
                     st.pyplot(plt)
               else:
                  st.error("The columns 'Fuel_Price' and/or 'CPI' do not exist in the dataset.")
               
               #Weekly Sales over time
               # Plot Weekly Sales
               plt.figure(figsize=(12, 6))
               walmart_data['Weekly_Sales'].plot(title='Weekly Sales Over Time')
               plt.xlabel('Time')
               plt.ylabel('Weekly Sales')
               plt.grid()
               with st.expander("Weekly Sales over time"):
                  st.pyplot(plt)

               # Perform Augmented Dickey-Fuller test
               sales_data = walmart_data['Weekly_Sales']
               result = adfuller(sales_data)
               p_value = result[1]

               # Display the p-value and stationarity result
               st.subheader("Augmented Dickey-Fuller Test")
               st.write(f"P-value: {p_value:.4f}")

               if p_value < 0.05:
                  st.write("The data is stationary.")
               else:
                  st.write("The data is not stationary.")

               #SARIMA 
                  # Convert the 'Date' column to a DateTimeIndex
               st.subheader("Sales Prediction Using SARIMA Model")
               # Check for required columns
               if 'Store' in walmart_data.columns:
                  # Calculate the total number of unique stores
                  unique_stores = walmart_data['Store'].unique()
                  num_stores = len(unique_stores)
                  
                  # Display the total number of stores
                  st.write(f"Total number of unique stores present in the uploaded file: {num_stores}")
               else:
                  st.warning("The uploaded file does not contain a 'store' column.")
               with st.expander("The Sales for next 12 Weeks"):
                     walmart_data['Date'] = pd.to_datetime(walmart_data['Date'],format="%d-%m-%Y")
                     # Drop any rows where Date is NaT
                     walmart_data = walmart_data.dropna(subset=['Date'])
                     
                     # Sort the data by Date
                     walmart_data = walmart_data.sort_values('Date')
                     # Define the SARIMA order
                     order = (1, 1, 1) 
                     seasonal_order = (1, 1, 1, 12)

                     # Set the 'Date' column as the index
                     walmart_data.set_index('Date', inplace=True)
                     if num_stores == 1:
                        store_id = unique_stores[0]
                        st.write(f"Only one store found: Store ID {store_id}. Proceeding with forecasting...")
                        store_data = walmart_data[walmart_data['Store'] == store_id]
                        # Ensure we have enough data for forecasting
                        if len(store_data) < 12:
                           st.error("Not enough data for forecasting. Need at least 12 weeks of data.")
                        else:
                           with st.spinner("Processing your data..."):
               
                              # Fit the SARIMA model
                              model = sm.tsa.statespace.SARIMAX(store_data['Weekly_Sales'], order=order, seasonal_order=seasonal_order)
                              model_fit = model.fit()

                              # Forecast future sales for 12 weeks
                              forecast = model_fit.forecast(steps=12)

                              # Calculate the last actual sales value
                              last_actual_sales = store_data['Weekly_Sales'].iloc[-1]

                              # Calculate percentage change from the last actual sales to the forecasted sales
                              percentage_changes = ((forecast - last_actual_sales) / last_actual_sales) * 100

                              # Display the forecasted values and their percentage changes
                              st.subheader(f"Forecasted Weekly Sales for Store ID: {store_data}")
                              results_df = pd.DataFrame({
                                 'Week': range(1, 13),
                                 'Predicted Sales': forecast,
                                 'Percentage Change (%)': percentage_changes,
                                 'Status': ['Gain' if pct > 0 else 'Loss' for pct in percentage_changes]
                              })

                              # Format predicted sales with dollar sign
                              results_df['Predicted Sales'] = results_df['Predicted Sales'].apply(lambda x: f"${x:.2f}")
                              results_df['Percentage Change (%)'] = results_df['Percentage Change (%)'].apply(lambda x: f"{x:.2f}")

                              st.write(results_df)

                              # Plot the original data
                              plt.figure(figsize=(10, 6))
                              plt.plot(store_data['Weekly_Sales'], label='Original Data', color='blue')

                              # Plot the forecasted data
                              forecast_index = pd.date_range(start=store_data.index[-1] + pd.DateOffset(weeks=1), periods=12, freq='W')
                              plt.plot(forecast_index, forecast, label='Forecasted Data', linestyle='--', marker='o', color='orange')

                              plt.title(f'Sales Forecast for Store ID: {store_data}')
                              plt.xlabel('Date')
                              plt.ylabel('Weekly Sales')
                              plt.legend()
                              plt.grid()
                              st.pyplot(plt)
                               # Show success message and balloons
                              st.success("Sales prediction completed! 🎉")
                              st.balloons() 

                     # Get user input for store ID
                     elif num_stores > 1:
                        store_id_input = st.number_input("Please enter the Store ID for forecasting sales:", min_value=int(walmart_data['Store'].min()), max_value=int(walmart_data['Store'].max()), step=1)

                        # Check if the entered store ID is valid
                        if st.button("Forecast Sales"):
                           if store_id_input not in walmart_data['Store'].unique():
                                 st.error("Invalid Store ID. Please enter a valid Store ID.")
                           else:
                               with st.spinner("Processing your data..."):
                                 # Filter data for the selected store
                                 store_data = walmart_data[walmart_data['Store'] == store_id_input]
                                 
                                 # Sort the data by the date index
                                 store_data.sort_index(inplace=True)

                                 # Fit the SARIMA model
                                 model = sm.tsa.statespace.SARIMAX(store_data['Weekly_Sales'], order=order, seasonal_order=seasonal_order)
                                 model_fit = model.fit()

                                 # Forecast future sales for 12 weeks
                                 forecast = model_fit.forecast(steps=12)

                                 # Calculate the last actual sales value
                                 last_actual_sales = store_data['Weekly_Sales'].iloc[-1]

                                 # Calculate percentage change from the last actual sales to the forecasted sales
                                 percentage_changes = ((forecast - last_actual_sales) / last_actual_sales) * 100

                                 # Display the forecasted values and their percentage changes
                                 st.subheader(f"Forecasted Weekly Sales for Store ID: {store_id_input}")
                                 results_df = pd.DataFrame({
                                    'Week': range(1, 13),
                                    'Predicted Sales': forecast,
                                    'Percentage Change (%)': percentage_changes,
                                    'Status': ['Gain' if pct > 0 else 'Loss' for pct in percentage_changes]
                                 })

                                 # Format predicted sales with dollar sign
                                 results_df['Predicted Sales'] = results_df['Predicted Sales'].apply(lambda x: f"${x:.2f}")
                                 results_df['Percentage Change (%)'] = results_df['Percentage Change (%)'].apply(lambda x: f"{x:.2f}")

                                 st.write(results_df)

                                 # Plot the original data
                                 plt.figure(figsize=(10, 6))
                                 plt.plot(store_data['Weekly_Sales'], label='Original Data', color='blue')

                                 # Plot the forecasted data
                                 forecast_index = pd.date_range(start=store_data.index[-1] + pd.DateOffset(weeks=1), periods=12, freq='W')
                                 plt.plot(forecast_index, forecast, label='Forecasted Data', linestyle='--', marker='o', color='orange')

                                 plt.title(f'Sales Forecast for Store ID: {store_id_input}')
                                 plt.xlabel('Date')
                                 plt.ylabel('Weekly Sales')
                                 plt.legend()
                                 plt.grid()
                                 st.pyplot(plt)
                                  # Show success message and balloons
                                 st.success("Sales prediction completed! 🎉")
                                 st.balloons()   
      except Exception as e:
         st.error(f"An error occurred: {e}")
         


   