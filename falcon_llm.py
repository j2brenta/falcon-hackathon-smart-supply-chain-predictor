import os
import openai
from dotenv import load_dotenv
from typing import Dict, List
import re
import logging

# Load environment variables from .env file
load_dotenv()

class FalconLLM:
    def __init__(self):
        self.ai71_base_url = os.getenv('AI71_BASE_URL', 'https://api.ai71.ai/v1/')
        self.ai71_api_key = os.getenv('AI71_API_KEY')
        
        if not self.ai71_api_key:
            raise ValueError("AI71_API_KEY not found in environment variables")

        self.client = openai.OpenAI(
            api_key=self.ai71_api_key,
            base_url=self.ai71_base_url,
        )
        self.model = "tiiuae/falcon-11b"

    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Helper method to get a response from the LLM.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM API call: {str(e)}")
            return ""

    def analyze_news(self, news: str) -> Dict:
        """
        Analyzes a news article to extract supply chain risks.
        
        Args:
            news (str): The news article text.
        
        Returns:
            Dict: Extracted risks and their potential impact.
        """
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in analyzing news for supply chain risks. Provide a concise analysis focusing on potential risks, their severity, and possible impacts on supply chains."},
            {"role": "user", "content": f"Analyze this news article for supply chain risks: {news}"}
        ]
        
        analysis = self._get_llm_response(messages)
        
        # Parse the analysis to extract key information
        lines = analysis.split('\n')
        risks = [line.strip() for line in lines if line.strip().startswith('-')]
        risk_level = "Medium"  # Default risk level
        
        return {
            "detected_risks": risks,
            "risk_level": risk_level,
            "potential_impact": analysis
        }

    def generate_risk_assessment(self, data: Dict) -> Dict:
            """
            Generates a risk assessment based on historical data and news analysis.
            
            Args:
                data (Dict): Processed supply chain data and news analysis results.
            
            Returns:
                Dict: Risk assessment including overall risk level and recommendations.
            """
            historical_risk = data.get("historical_risk", "Unknown")
            news_risk = data.get("news_risk", "Unknown")
            detected_risks = data.get("detected_risks", [])
            
            prompt = f"""
            Based on the following information, provide a comprehensive supply chain risk assessment:
            
            Historical Risk Level: {historical_risk}
            News-based Risk Level: {news_risk}
            Detected Risks: {', '.join(detected_risks)}
            
            Include in your assessment:
            1. An overall risk level (Low, Medium, or High)
            2. A brief explanation of the risk assessment
            3. Key risk factors
            4. 3-5 actionable recommendations for risk mitigation
            
            Format your response with clear headers for each section.
            """
            
            messages = [
                {"role": "system", "content": "You are an AI assistant specialized in supply chain risk assessment and mitigation strategies."},
                {"role": "user", "content": prompt}
            ]
            
            assessment = self._get_llm_response(messages)

            logging.info(f"Generated risk assessment: {assessment}")
            
            # Parse the assessment using regex to be more flexible
            overall_risk_level = re.search(r"Overall Risk Level:?\s*(\w+)", assessment, re.IGNORECASE)
            overall_risk_level = overall_risk_level.group(1) if overall_risk_level else "Medium"
            
            explanation = re.search(r"Brief Explanation[:\n]+(.*?)(?=\n\d+\.|\Z)", assessment, re.DOTALL | re.IGNORECASE)
            explanation = explanation.group(1).strip() if explanation else ""
            
            risk_factors = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|\Z)", assessment, re.DOTALL)
            
            recommendations = re.findall(r"[a-z]\)\s*(.*?)(?=\n[a-z]\)|\Z)", assessment, re.DOTALL)
            
            return {
                "overall_risk_level": overall_risk_level,
                "assessment": explanation,
                "risk_factors": risk_factors,
                "recommendations": recommendations
            }