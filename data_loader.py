import pandas as pd
from typing import List, Dict

class DataLoader:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.mock_news_data = [
            "Severe flooding in Taiwan disrupts semiconductor production",
            "Trade tensions escalate between China and the United States",
            "Major cyberattack targets global shipping companies",
            "Labor strikes at key European ports cause shipment delays",
            "Unexpected surge in oil prices impacts global transportation costs"
        ]

    def load_historical_data(self) -> pd.DataFrame:
        """
        Loads historical supply chain data from a CSV file.
        
        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            print(f"Successfully loaded {len(df)} records from {self.csv_file_path}")
            return df
        except FileNotFoundError:
            print(f"Error: File {self.csv_file_path} not found.")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Error: File {self.csv_file_path} is empty.")
            return pd.DataFrame()
        except pd.errors.ParserError:
            print(f"Error: Unable to parse {self.csv_file_path}. Check if it's a valid CSV.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return pd.DataFrame()

    def load_mock_news_data(self) -> List[str]:
        """
        Loads pre-defined mock news data.
        
        Returns:
            List[str]: A list of mock news headlines.
        """
        return self.mock_news_data

    def get_data_summary(self) -> Dict:
        """
        Provides a summary of the loaded historical data.
        
        Returns:
            Dict: A dictionary containing summary statistics of the data.
        """
        df = self.load_historical_data()
        if df.empty:
            return {"error": "No data available"}
        
        return {
            "total_records": len(df),
            "date_range": (df['Date'].min(), df['Date'].max()),
            "unique_products": df['Product_ID'].nunique(),
            "unique_suppliers": df['Supplier_ID'].nunique(),
            "avg_delay_days": df['Days_Delayed'].mean()
        }
