# src/api/routes.py
from flask import Flask, request, jsonify, render_template_string
from src.api.schemas import ChatRequest, ChatResponse, FeedbackRequest, StatsResponse
from src.services.intelligent_assistant import IntelligentAssistant
from src.utils.logger import setup_logger
from src.utils.sqlite_database import SQLiteDataService
from src.services.sqlite_excel_scheduler import sqlite_excel_scheduler
import os

logger = setup_logger(__name__)

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Intelligent Customer Assistant</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .chat-container { 
            border: 2px solid #ddd; 
            border-radius: 10px; 
            padding: 20px; 
            height: 400px; 
            overflow-y: auto; 
            margin-bottom: 20px; 
            background-color: white;
        }
        .user-message { 
            background: #007bff; 
            color: white; 
            padding: 10px; 
            border-radius: 10px; 
            margin: 5px 0; 
            text-align: right; 
            max-width: 80%;
            margin-left: auto;
        }
        .bot-message { 
            background: #f1f1f1; 
            padding: 10px; 
            border-radius: 10px; 
            margin: 5px 0;
            max-width: 80%;
        }
        .confidence { 
            font-size: 0.8em; 
            color: #666; 
        }
        .input-container { 
            display: flex; 
            gap: 10px; 
            margin-bottom: 20px;
        }
        input[type="text"] { 
            flex: 1; 
            padding: 12px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            font-size: 16px;
        }
        button { 
            padding: 12px 24px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
        .stats-container {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
        }
        .intent-badge {
            display: inline-block;
            padding: 2px 8px;
            background: #28a745;
            color: white;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Intelligent Customer Assistant</h1>
        <p>AI-powered customer support with context awareness</p>
    </div>
    
    <div class="chat-container" id="chat">
        <div class="bot-message">
            <strong>Hello! I'm your AI customer assistant. 👋</strong><br>
            I can help you with: password resets, order status, billing issues, product information, 
            technical support, returns & refunds, and account management.
        </div>
    </div>
    
    <div class="input-container">
        <input type="text" id="userInput" placeholder="Type your question here... (e.g., 'I forgot my password')" onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <div class="stats-container">
        <h3>📊 Performance Statistics</h3>
        <div id="stats">Loading statistics...</div>
    </div>

    <script>
        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message to chat
            const chat = document.getElementById('chat');
            const userMsg = document.createElement('div');
            userMsg.className = 'user-message';
            userMsg.textContent = message;
            chat.appendChild(userMsg);
            
            // Clear input and disable
            input.value = '';
            input.disabled = true;
            
            // Show typing indicator
            const typingMsg = document.createElement('div');
            typingMsg.className = 'bot-message';
            typingMsg.innerHTML = '<em>🤖 Thinking...</em>';
            chat.appendChild(typingMsg);
            chat.scrollTop = chat.scrollHeight;
            
            // Send to backend
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message, user_id: 'web_user'})
            })
            .then(response => response.json())
            .then(data => {
                // Remove typing indicator
                chat.removeChild(typingMsg);
                
                // Add bot response to chat
                const botMsg = document.createElement('div');
                botMsg.className = 'bot-message';
                botMsg.innerHTML = `
                    <strong>${data.response}</strong><br>
                    <span class="confidence">
                        Intent: ${data.predicted_intent} 
                        <span class="intent-badge">${(data.confidence * 100).toFixed(1)}%</span>
                    </span>
                `;
                chat.appendChild(botMsg);
                chat.scrollTop = chat.scrollHeight;
                
                // Re-enable input
                input.disabled = false;
                input.focus();
                
                // Update stats
                updateStats();
            })
            .catch(error => {
                console.error('Error:', error);
                chat.removeChild(typingMsg);
                const errorMsg = document.createElement('div');
                errorMsg.className = 'bot-message';
                errorMsg.innerHTML = '<strong>❌ Sorry, I encountered an error. Please try again.</strong>';
                chat.appendChild(errorMsg);
                input.disabled = false;
                input.focus();
            });
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function updateStats() {
            fetch('/stats')
            .then(response => response.json())
            .then(stats => {
                document.getElementById('stats').innerHTML = `
                    <strong>Total Queries:</strong> ${stats.total_queries_processed} | 
                    <strong>Resolution Rate:</strong> ${stats.resolution_rate} | 
                    <strong>Avg Confidence:</strong> ${stats.average_confidence}
                    ${stats.active_users ? `| <strong>Active Users:</strong> ${stats.active_users}` : ''}
                `;
            })
            .catch(error => {
                document.getElementById('stats').innerHTML = 'Error loading statistics';
            });
        }
        
        // Initial stats load
        updateStats();
        
        // Focus input on load
        document.getElementById('userInput').focus();
    </script>
</body>
</html>
'''

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    assistant_service = IntelligentAssistant()
    
    logger.info("Intelligent Assistant initialized successfully")
    
    @app.route('/')
    def home():
        """Main web interface"""
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy", 
            "service": "Customer Assistant API",
            "version": "1.0.0"
        })
    
    @app.route('/chat', methods=['POST'])
    def chat():
        """Chat endpoint"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
                
            chat_request = ChatRequest(**data)
            
            result = assistant_service.process_message(
                message=chat_request.message,
                user_id=chat_request.user_id
            )
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/stats', methods=['GET'])
    def get_stats():
        """Get performance statistics"""
        try:
            stats = assistant_service.get_performance_stats()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error in stats endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/feedback', methods=['POST'])
    def submit_feedback():
        """Submit feedback for model improvement"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
                
            feedback_request = FeedbackRequest(**data)
            
            # Store feedback for retraining
            assistant_service.feedback_data.append({
                'query': feedback_request.query,
                'predicted_intent': feedback_request.predicted_intent,
                'correct_intent': feedback_request.correct_intent,
                'user_id': feedback_request.user_id
            })
            
            return jsonify({"status": "feedback recorded"})
            
        except Exception as e:
            logger.error(f"Error in feedback endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/customer/<customer_id>', methods=['GET'])
    def get_customer_info(customer_id):
        """Get customer information"""
        try:
            customer = SQLiteDataService.get_customer_info(customer_id)
            if not customer:
                return jsonify({"error": "Customer not found"}), 404
            return jsonify(customer)
        except Exception as e:
            logger.error(f"Error getting customer info: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/order/<order_id>', methods=['GET'])
    def get_order_status(order_id):
        """Get order status"""
        try:
            order = SQLiteDataService.get_order_status(order_id)
            if not order:
                return jsonify({"error": "Order not found"}), 404
            return jsonify(order)
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/products/search', methods=['GET'])
    def search_products():
        """Search products"""
        try:
            search_term = request.args.get('q', '')
            category_id = request.args.get('category_id', type=int)
            
            if not search_term:
                return jsonify({"error": "Search term required"}), 400
                
            products = SQLiteDataService.search_products(search_term, category_id)
            return jsonify({"products": products})
        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/export/excel', methods=['POST'])
    def export_excel():
        """Trigger Excel export"""
        try:
            data = request.get_json() or {}
            export_type = data.get('type', 'both')  # realtime, analytics, or both
            
            success = sqlite_excel_scheduler.export_now(export_type)
            if success:
                return jsonify({"status": "export completed"})
            else:
                return jsonify({"error": "export failed"}), 500
        except Exception as e:
            logger.error(f"Error in Excel export: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app, assistant_service