from typing import Dict, List
import pandas as pd

class ContextManager:
    def __init__(self):
        self.user_contexts = {}
        self.conversation_history = []
    
    def get_user_context(self, user_id: str) -> Dict:
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                'last_intent': None,
                'last_entities': {},
                'conversation_count': 0,
                'session_start': pd.Timestamp.now()
            }
        return self.user_contexts[user_id]
    
    def update_context(self, user_id: str, intent: str, entities: Dict):
        context = self.get_user_context(user_id)
        context['last_intent'] = intent
        context['last_entities'] = entities
        context['conversation_count'] += 1
    
    def add_conversation(self, user_id: str, query: str, intent: str, response: str, confidence: float):
        conversation_entry = {
            'user_id': user_id,
            'timestamp': pd.Timestamp.now(),
            'query': query,
            'intent': intent,
            'response': response,
            'confidence': confidence
        }
        self.conversation_history.append(conversation_entry)
    
    def get_conversation_history(self, user_id: str = None) -> List[Dict]:
        if user_id:
            return [conv for conv in self.conversation_history if conv['user_id'] == user_id]
        return self.conversation_history
    
    def clear_user_context(self, user_id: str):
        if user_id in self.user_contexts:
            del self.user_contexts[user_id]