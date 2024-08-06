import pandas as pd
from typing import Dict, List
from falcon_llm import FalconLLM

class Predictor:
    def __init__(self, llm: FalconLLM):
        self.llm = llm
        self.risk_levels = {"Low": 1, "Medium": 2, "High": 3}

    def predict_risk(self, historical_data: pd.DataFrame, news_data: List[str]) -> Dict:
        """
        Predicts supply chain risk based on historical data and news analysis.
        
        Args:
            historical_data (pd.DataFrame): Processed historical supply chain data.
            news_data (List[str]): List of relevant news articles.
        
        Returns:
            Dict: Predicted risk assessment including overall risk level, factors, and recommendations.
        """
        historical_risk = self._analyze_historical_data(historical_data)
        news_risk = self._analyze_news_data(news_data)
        
        combined_data = {
            "historical_risk": historical_risk["risk_level"],
            "news_risk": news_risk["risk_level"],
            "detected_risks": news_risk["detected_risks"] + historical_risk["risk_factors"]
        }
        
        final_assessment = self.llm.generate_risk_assessment(combined_data)
        
        return {
            "overall_risk_level": final_assessment["overall_risk_level"],
            "risk_factors": list(set(final_assessment["risk_factors"])),  # Remove duplicates
            "assessment": final_assessment["assessment"],
            "recommendations": final_assessment["recommendations"],
            "historical_insights": historical_risk["insights"],
            "news_insights": news_risk["insights"]
        }

    def _analyze_historical_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyzes historical data to determine risk levels and factors.
        
        Args:
            data (pd.DataFrame): Processed historical supply chain data.
        
        Returns:
            Dict: Historical risk assessment.
        """
        # Calculate basic risk metrics
        delay_rate = (data['Is_Delayed'] == 1).mean()
        quantity_discrepancy_rate = (data['Quantity_Difference'] != 0).mean()
        
        # Determine risk level based on delay rate
        if delay_rate > 0.2:
            risk_level = "High"
        elif delay_rate > 0.1:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Identify top risk factors
        risk_factors = []
        if delay_rate > 0.1:
            risk_factors.append("frequent delays")
        if quantity_discrepancy_rate > 0.1:
            risk_factors.append("quantity discrepancies")
        
        # Generate insights
        insights = [
            f"Historical delay rate: {delay_rate:.2%}",
            f"Quantity discrepancy rate: {quantity_discrepancy_rate:.2%}",
            f"Most common transportation mode: {data['Transportation_Mode'].mode().iloc[0]}",
            f"Average political stability index: {data['Political_Stability_Index'].mean():.2f}"
        ]
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "insights": insights
        }

    def _analyze_news_data(self, news_articles: List[str]) -> Dict:
        """
        Analyzes news data to determine risk levels and factors.
        
        Args:
            news_articles (List[str]): List of relevant news articles.
        
        Returns:
            Dict: News-based risk assessment.
        """
        all_risks = []
        risk_levels = []
        insights = []
        
        for article in news_articles:
            analysis = self.llm.analyze_news(article)
            all_risks.extend(analysis["detected_risks"])
            risk_levels.append(self.risk_levels[analysis["risk_level"]])
            insights.append(analysis["potential_impact"])
        
        # Determine overall risk level from news
        avg_risk_level = sum(risk_levels) / len(risk_levels)
        if avg_risk_level > 2:
            news_risk_level = "High"
        elif avg_risk_level > 1.5:
            news_risk_level = "Medium"
        else:
            news_risk_level = "Low"
        
        return {
            "risk_level": news_risk_level,
            "detected_risks": list(set(all_risks)),  # Remove duplicates
            "insights": insights
        }
