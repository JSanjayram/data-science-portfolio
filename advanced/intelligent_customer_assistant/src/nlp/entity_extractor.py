import re
from typing import Dict, List

class EntityExtractor:
    """Entity extraction service"""
    
    def __init__(self):
        self.patterns = {
            'order_number': r'\b\d{5,}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        
        self.urgency_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'right now']
        self.product_keywords = ['phone', 'smartphone', 'laptop', 'tablet', 'device']
    
    def extract_entities(self, text: str) -> Dict:
        """Extract entities from text"""
        entities = {}
        text_lower = text.lower()
        
        # Extract patterns
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = matches[0]
        
        # Extract urgency
        if any(word in text_lower for word in self.urgency_keywords):
            entities['urgency'] = 'high'
        
        # Extract product type
        for product in self.product_keywords:
            if product in text_lower:
                entities['product_type'] = product
                break
        
        return entities