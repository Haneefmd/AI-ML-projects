import streamlit as st
# Set the page layout to wide
st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")
#####
def app():
    st.markdown("""
        <style>
        .animated-header {
            animation: fadeIn 2s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Use the animated class in your header
    st.markdown('<h1 class="animated-header">Welcome to the Sales Prediction Dashboard!</h1>', unsafe_allow_html=True)

    # Add a subtitle
    st.subheader("Unlock the Power of Sales Forecasting")

    # Add a brief introduction
    st.write("""
    Sales prediction is a crucial aspect of business strategy that helps organizations anticipate future sales trends based on historical data. 
    By leveraging advanced analytics and machine learning techniques, businesses can make informed decisions that drive growth and efficiency.
    """)

    # Add an image (replace 'sales_prediction_image.jpg' with your actual image path)
    st.image(r"C:\Users\Faheema\Downloads\Sales project\sales_prediction_image.jpeg", caption="Sales Prediction Visualization", use_container_width=True)

    # Importance of Sales Prediction
    st.header("Why is Sales Prediction Important?")
    st.write("""
    1. **Informed Decision-Making**: Accurate sales forecasts enable businesses to make data-driven decisions regarding inventory, staffing, and marketing strategies.
    2. **Resource Optimization**: By predicting sales accurately, companies can optimize their resources, reducing waste and increasing profitability.
    3. **Customer Satisfaction**: Anticipating customer demand helps in maintaining adequate stock levels, ensuring that customer needs are met promptly.
    4. **Strategic Planning**: Businesses can plan for seasonal fluctuations and market trends, positioning themselves to capitalize on opportunities.
    """)

    # How the App Works
    st.header("How This App Works")
    st.write("""
    Our Sales Prediction Dashboard utilizes the SARIMAX (Seasonal Autoregressive Integrated Moving Average eXogenous variable) model to analyze your sales data and provide accurate forecasts. 
    Here's how you can get started:
    1. **Upload Your Data**: Start by uploading your historical sales data in CSV format.
    2. **Select Features**: Choose the relevant features that you want to include in the prediction model.
    3. **Get Predictions**: Click the predict button and receive your sales forecasts along with visualizations.
    """)

    # Features of the App
    st.header("Key Features")
    st.write("""
    - **User -Friendly Interface**: Easy navigation and intuitive design for seamless user experience.
    - **SARIMAX Model**: Utilize the powerful SARIMAX model for accurate sales forecasting based on historical data.
    - **Data Visualization**: Visualize your sales trends and predictions through interactive charts and graphs.
    - **Feedback Mechanism**: Provide feedback to help us improve the app.
    """)

    # Call to Action
    st.header("Ready to Get Started?")
    st.write("""
    Upload your sales data and start predicting your future sales today! Navigate to the 'Sales Prediction' section in the menu.
    """)

    # Footer
    st.markdown("---")
    st.write("Developed by Md & TEAM")
    st.write("For inquiries, contact us at: [mdhaneef833@gmail.com]")