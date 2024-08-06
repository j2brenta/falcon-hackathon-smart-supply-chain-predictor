import streamlit as st
import pandas as pd
import logging
from data_loader import DataLoader
from data_processor import DataProcessor
from falcon_llm import FalconLLM
from predictor import Predictor
from dashboard import Dashboard

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MainApp:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.loader = DataLoader(data_file)
        self.processor = DataProcessor()
        self.llm = FalconLLM()
        self.predictor = Predictor(self.llm)
        self.dashboard = Dashboard()

    def load_and_process_data(self) -> tuple:
        """
        Loads and processes the historical supply chain data and news data.
        
        Returns:
            tuple: Processed historical data and news data.
        """
        try:
            logging.info("Loading historical data...")
            raw_data = self.loader.load_historical_data()
            
            logging.info("Processing historical data...")
            processed_data = self.processor.process_historical_data(raw_data)
            
            logging.info("Loading news data...")
            news_data = self.loader.load_mock_news_data()
            
            return processed_data, news_data
        except Exception as e:
            logging.error(f"Error in loading or processing data: {str(e)}")
            st.error(f"An error occurred while loading or processing data: {str(e)}")
            return pd.DataFrame(), []

    def generate_prediction(self, historical_data: pd.DataFrame, news_data: list) -> dict:
        """
        Generates risk prediction based on historical and news data.
        
        Args:
            historical_data (pd.DataFrame): Processed historical supply chain data.
            news_data (list): List of news articles.
        
        Returns:
            dict: Risk prediction results.
        """
        try:
            logging.info("Generating risk prediction...")
            return self.predictor.predict_risk(historical_data, news_data)
        except Exception as e:
            logging.error(f"Error in generating prediction: {str(e)}")
            st.error(f"An error occurred while generating the prediction: {str(e)}")
            return {}

    def run(self):
        """
        Runs the main application flow.
        """
        st.sidebar.title("Supply Chain Risk Predictor")
        st.sidebar.info("This application predicts supply chain risks based on historical data and recent news.")
        
        if st.sidebar.button("Run Prediction"):
            with st.spinner("Loading and processing data..."):
                historical_data, news_data = self.load_and_process_data()
                
                if historical_data.empty:
                    st.warning("No historical data available. Please check your data source.")
                    return
                
            with st.spinner("Generating risk prediction..."):
                risk_prediction = self.generate_prediction(historical_data, news_data)
                
                if not risk_prediction:
                    st.warning("Unable to generate risk prediction. Please try again.")
                    return
            
            logging.info("Displaying dashboard...")
            self.dashboard.run_dashboard(risk_prediction, historical_data)
        
        else:
            st.info("Click 'Run Prediction' in the sidebar to start the analysis.")

if __name__ == "__main__":
    app = MainApp("supply_chain_data.csv")
    app.run()