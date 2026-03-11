import streamlit as st
import pandas as pd
from inference import generate_llm_recommendation_text, get_demand_predictions, generate_business_insights

#configure page settings
st.set_page_config(page_title="E-Cart Intelligence Dashboard", layout="wide")

#main title
st.title("E-Cart Intelligent Platform")
st.write("Machine learning and AI-driven insights for product recommendations and demand prediction.")

#create tabs for the different business functions
tab1, tab2, tab3 = st.tabs(["Product Recommendations", "Demand Prediction", "Business Insights"])

#tab 1: recommendation system
with tab1:
    st.header("Customer Recommendations")
    st.write("Generate personalized product recommendations for a specific user.")
    
    #user input for customer id
    user_id_input = st.number_input("Enter User ID:", min_value=1, step=1, value=102)
    
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            #fetch llm formatted recommendations
            recommendation_text = generate_llm_recommendation_text(user_id_input)
            st.markdown(recommendation_text)

#tab 2: demand prediction
with tab2:
    st.header("Next Week Demand Forecast")
    st.write("Forecasted high-demand products for inventory planning and supply chain management.")
    
    if st.button("Run Demand Forecast"):
        with st.spinner("Calculating forecast..."):
            #fetch the forecasted dataframe
            forecast_df = get_demand_predictions()
            
            #display as a clean table
            st.dataframe(forecast_df, use_container_width=True)
            
            #render a bar chart for visual analysis
            st.write("Demand Visualization")
            st.bar_chart(data=forecast_df.set_index('product_name')['predicted_demand'])

#tab 3: business insights report
with tab3:
    st.header("Executive Business Insights")
    st.write("AI-generated analysis of platform performance, inventory needs, and customer segments.")
    
    if st.button("Generate Insights Report"):
        with st.spinner("Analyzing platform data and generating report..."):
            #fetch llm generated business insights
            insights_text = generate_business_insights()
            st.markdown(insights_text)