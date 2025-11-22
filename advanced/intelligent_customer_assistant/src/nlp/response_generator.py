from typing import Dict, List

class ResponseGenerator:
    def __init__(self):
        self.templates = {
            'password_reset': "🔐 You can reset your password by visiting our password recovery page...",
            'order_status': "📦 I can check your order status! Please provide your order number...",
            'billing_issue': "💳 I understand you're having billing issues...",
            'product_info': "📱 I'd be happy to provide product information...",
            'technical_support': "🛠️ I'm sorry you're experiencing technical issues...",
            'return_refund': "📮 Our return policy allows returns within 30 days...",
            'account_management': "👤 I can help with account management...",
            'default': "❓ I'm not quite sure I understand..."
        }
    
    def generate_response(self, intent: str, entities: Dict = None, context: Dict = None) -> str:
        base_response = self.templates.get(intent, self.templates['default'])
        
        if entities and intent == 'order_status' and 'order_number' in entities:
            return f"📦 I found order #{entities['order_number']}! {base_response}"
        
        if entities and 'urgency' in entities:
            return f"🚨 I understand this is urgent! {base_response}"
        
        return base_response
    
    def add_template(self, intent: str, template: str):
        self.templates[intent] = template