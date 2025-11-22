from typing import Dict, List
import pandas as pd
import os

class FeedbackService:
    def __init__(self, feedback_file: str = 'data/feedback.csv'):
        self.feedback_file = feedback_file
        self.feedback_data = []
    
    def add_feedback(self, query: str, predicted_intent: str, correct_intent: str, user_id: str = None):
        feedback_entry = {
            'timestamp': pd.Timestamp.now(),
            'query': query,
            'predicted_intent': predicted_intent,
            'correct_intent': correct_intent,
            'user_id': user_id or 'anonymous'
        }
        self.feedback_data.append(feedback_entry)
    
    def save_feedback(self):
        if self.feedback_data:
            df = pd.DataFrame(self.feedback_data)
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            df.to_csv(self.feedback_file, index=False)
    
    def load_feedback(self) -> pd.DataFrame:
        if os.path.exists(self.feedback_file):
            return pd.read_csv(self.feedback_file)
        return pd.DataFrame()
    
    def get_feedback_stats(self) -> Dict:
        if not self.feedback_data:
            return {"message": "No feedback data available"}
        
        df = pd.DataFrame(self.feedback_data)
        accuracy = (df['predicted_intent'] == df['correct_intent']).mean()
        
        return {
            'total_feedback': len(self.feedback_data),
            'accuracy': f"{accuracy:.2%}",
            'most_confused_intent': df[df['predicted_intent'] != df['correct_intent']]['predicted_intent'].mode().iloc[0] if len(df) > 0 else None
        }