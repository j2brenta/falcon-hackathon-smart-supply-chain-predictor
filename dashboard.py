import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Supply Chain Risk Predictor", layout="wide")
        st.title("Supply Chain Risk Predictor Dashboard")

    def display_risk_assessment(self, risk: Dict):
        """
        Displays the overall risk assessment.
        
        Args:
            risk (Dict): The risk prediction results.
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Risk Level", risk['overall_risk_level'])
        
        with col2:
            st.write("Key Risk Factors:")
            for factor in risk['risk_factors']:
                st.write(f"• {factor}")
        
        with col3:
            st.write("Top Recommendations:")
            for rec in risk['recommendations'][:3]:  # Display top 3 recommendations
                st.write(f"• {rec}")
        
        st.write("### Detailed Assessment")
        st.write(risk['assessment'])

    def plot_historical_data(self, data: pd.DataFrame):
        """
        Plots key metrics from historical data.
        
        Args:
            data (pd.DataFrame): Processed historical supply chain data.
        """
        st.write("### Historical Data Insights")

        # Delay Rate Over Time
        delay_rate = data.groupby('Date')['Is_Delayed'].mean().reset_index()
        fig_delay = px.line(delay_rate, x='Date', y='Is_Delayed', title='Delay Rate Over Time')
        st.plotly_chart(fig_delay)

        # Quantity Discrepancy Over Time
        qty_discrepancy = data.groupby('Date')['Quantity_Difference'].mean().reset_index()
        fig_qty = px.line(qty_discrepancy, x='Date', y='Quantity_Difference', title='Average Quantity Discrepancy Over Time')
        st.plotly_chart(fig_qty)

        # Top Suppliers by Delay Rate
        top_suppliers = data.groupby('Supplier_Name')['Is_Delayed'].mean().sort_values(ascending=False).head(10)
        fig_suppliers = px.bar(top_suppliers, title='Top 10 Suppliers by Delay Rate')
        st.plotly_chart(fig_suppliers)

    def display_news_insights(self, news_insights: list):
        """
        Displays insights extracted from news articles.
        
        Args:
            news_insights (list): List of insights from news analysis.
        """
        st.write("### News Analysis Insights")
        for insight in news_insights:
            st.info(insight)

    def run_dashboard(self, risk_prediction: Dict, historical_data: pd.DataFrame):
        """
        Runs the main dashboard application.
        
        Args:
            risk_prediction (Dict): The risk prediction results.
            historical_data (pd.DataFrame): Processed historical supply chain data.
        """
        self.display_risk_assessment(risk_prediction)
        self.plot_historical_data(historical_data)
        self.display_news_insights(risk_prediction['news_insights'])

        # Interactive Data Explorer
        st.write("### Historical Data Explorer")
        selected_columns = st.multiselect(
            "Select columns to view",
            options=historical_data.columns,
            default=["Date", "Product_Name", "Supplier_Name", "Quantity_Ordered", "Is_Delayed"]
        )
        st.dataframe(historical_data[selected_columns])