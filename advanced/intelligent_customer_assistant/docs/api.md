# API Documentation

## Endpoints

### Health Check
- **GET** `/health`
- Returns service status

### Chat
- **POST** `/chat`
- Process user queries
- Body: `{"message": "string", "user_id": "string"}`

### Statistics
- **GET** `/stats`
- Get performance metrics

### Feedback
- **POST** `/feedback`
- Submit model feedback
- Body: `{"query": "string", "predicted_intent": "string", "correct_intent": "string"}`

## Response Format

All responses follow JSON format with appropriate HTTP status codes.

## Authentication

Currently no authentication required for development.