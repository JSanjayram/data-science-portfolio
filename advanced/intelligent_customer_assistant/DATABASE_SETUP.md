# Database Setup Guide

## Overview
This project uses two PostgreSQL databases:
1. **Company Database** - Training data, intents, entities, company information
2. **E-commerce Database** - Customer data, orders, products, analytics

## Quick Setup

### 1. Automated Setup (Recommended)
```bash
python setup_project.py
```

### 2. Manual Setup

#### Start Docker Containers
```bash
docker-compose -f docker-compose-full.yml up -d
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run Application
```bash
python run.py
```

## Database Schema

### Company Database Tables
- `company_info` - Company knowledge base
- `intents` - Chatbot intents and responses
- `entities` - Named entity definitions
- `training_data` - Training examples

### E-commerce Database Tables
- `customers` - Customer information
- `products` - Product catalog
- `orders` - Order management
- `order_items` - Order line items
- `shipments` - Shipping tracking
- `support_tickets` - Customer support
- `product_reviews` - Customer reviews
- `customer_interactions` - Chat logs

## Real-time Excel Exports

### Automatic Exports
- **Real-time data**: Every 30 minutes
- **Analytics data**: Every hour
- **Location**: `data/exports/`

### Manual Export
```bash
curl -X POST http://localhost:5000/export/excel \
  -H "Content-Type: application/json" \
  -d '{"type": "both"}'
```

### Export Types
- `realtime` - Current operational data
- `analytics` - EDA and analysis data
- `both` - All data types

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /chat` - Chat with assistant
- `GET /stats` - Performance statistics

### Data Endpoints
- `GET /customer/<customer_id>` - Customer information
- `GET /order/<order_id>` - Order status
- `GET /products/search?q=<term>` - Product search
- `POST /export/excel` - Trigger Excel export

## Sample Data

The system includes comprehensive sample data:
- 3 customers with order history
- 5 products across different categories
- Multiple orders with different statuses
- Support tickets and reviews
- Customer interaction logs

## Excel Export Sheets

### Real-time Export
- `customers` - Customer list
- `orders` - Order details
- `products` - Product catalog
- `shipments` - Shipping status
- `interactions` - Chat logs
- `support_tickets` - Support cases
- `sales_summary` - Daily sales

### Analytics Export
- `customer_metrics` - Customer analytics
- `product_performance` - Product metrics
- `monthly_trends` - Trend analysis
- `intent_analysis` - Chatbot performance

## Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

### Database URLs
- Company: `postgresql://assistant_user:assistant_pass@localhost:5432/company_db`
- E-commerce: `postgresql://assistant_user:assistant_pass@localhost:5432/ecommerce_db`

## Troubleshooting

### Database Connection Issues
1. Ensure Docker containers are running:
   ```bash
   docker ps
   ```

2. Check container logs:
   ```bash
   docker logs assistant_postgres
   ```

3. Test connection manually:
   ```bash
   psql -h localhost -p 5432 -U assistant_user -d company_db
   ```

### Excel Export Issues
1. Check export directory exists: `data/exports/`
2. Verify database connections
3. Check application logs

## Production Considerations

### Security
- Change default passwords
- Use environment variables for credentials
- Enable SSL for database connections
- Implement proper authentication

### Performance
- Configure connection pooling
- Add database indexes
- Monitor query performance
- Set up database backups

### Scaling
- Use read replicas for analytics
- Implement caching with Redis
- Consider database partitioning
- Monitor resource usage