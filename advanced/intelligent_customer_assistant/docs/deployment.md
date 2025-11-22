# Deployment Guide

## Docker Deployment

```bash
docker build -t intelligent-assistant .
docker run -p 5000:5000 intelligent-assistant
```

## Production Deployment

1. Set environment variables
2. Use production config
3. Set up monitoring
4. Configure load balancer

## Environment Variables

- `CONFIG_PATH`: Path to configuration file
- `LOG_LEVEL`: Logging level
- `PORT`: Application port

## Monitoring

- Health checks on `/health`
- Metrics collection enabled
- Log aggregation configured