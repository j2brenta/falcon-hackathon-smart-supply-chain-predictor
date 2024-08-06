import pandas as pd
import numpy as np
from typing import List

class DataProcessor:
    def __init__(self):
        pass

    def process_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and prepares historical data for analysis.
        
        Args:
            data (pd.DataFrame): Raw historical supply chain data.
        
        Returns:
            pd.DataFrame: Processed data.
        """
        # Create a copy to avoid modifying the original dataframe
        df = data.copy()
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle missing values
        df['Quantity_Received'].fillna(df['Quantity_Ordered'], inplace=True)
        df['Days_Delayed'].fillna(0, inplace=True)
        
        # Create a binary column for delayed shipments
        df['Is_Delayed'] = df['Days_Delayed'].apply(lambda x: 1 if x > 0 else 0)
        
        # Calculate the difference between ordered and received quantities
        df['Quantity_Difference'] = df['Quantity_Ordered'] - df['Quantity_Received']
        
        # Convert categorical variables to numerical
        df['Transportation_Mode'] = pd.Categorical(df['Transportation_Mode']).codes
        df['Weather_Condition'] = pd.Categorical(df['Weather_Condition']).codes
        
        # Handle 'N/A' in Port_Congestion_Level
        df['Port_Congestion_Level'] = df['Port_Congestion_Level'].replace('N/A', np.nan)
        df['Port_Congestion_Level'] = pd.Categorical(df['Port_Congestion_Level'], 
                                                     categories=['Low', 'Medium', 'High']).codes
        
        return df

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts relevant features from the processed data.
        
        Args:
            data (pd.DataFrame): Processed historical supply chain data.
        
        Returns:
            pd.DataFrame: DataFrame with extracted features.
        """
        features = pd.DataFrame()
        
        # Time-based features
        features['Year'] = data['Date'].dt.year
        features['Month'] = data['Date'].dt.month
        features['DayOfWeek'] = data['Date'].dt.dayofweek
        
        # Supplier performance features
        features['Supplier_Delay_Rate'] = data.groupby('Supplier_ID')['Is_Delayed'].transform('mean')
        features['Supplier_Quantity_Difference_Avg'] = data.groupby('Supplier_ID')['Quantity_Difference'].transform('mean')
        
        # Product features
        features['Product_Delay_Rate'] = data.groupby('Product_ID')['Is_Delayed'].transform('mean')
        features['Product_Quantity_Difference_Avg'] = data.groupby('Product_ID')['Quantity_Difference'].transform('mean')
        
        # Other relevant features
        features['Transportation_Mode'] = data['Transportation_Mode']
        features['Weather_Condition'] = data['Weather_Condition']
        features['Political_Stability_Index'] = data['Political_Stability_Index']
        features['Port_Congestion_Level'] = data['Port_Congestion_Level']
        
        # Target variable
        features['Is_Delayed'] = data['Is_Delayed']
        
        return features

    def get_feature_importance(self, features: pd.DataFrame) -> List[tuple]:
        """
        Calculates a simple feature importance based on correlation with the target variable.
        
        Args:
            features (pd.DataFrame): Extracted features.
        
        Returns:
            List[tuple]: Sorted list of feature importances.
        """
        correlations = features.corr()['Is_Delayed'].abs().sort_values(ascending=False)
        return list(correlations.items())