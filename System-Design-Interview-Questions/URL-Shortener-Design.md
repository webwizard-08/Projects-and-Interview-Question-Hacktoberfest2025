# 🔗 URL Shortener (TinyURL) - System Design

## 📋 Problem Statement

Design a URL shortener service like TinyURL that can:
- Shorten long URLs to short URLs
- Redirect short URLs to original URLs
- Handle high traffic (100M URLs per day)
- Provide analytics and statistics
- Ensure high availability and scalability

## 🎯 Requirements Gathering

### Functional Requirements
- **URL Shortening**: Convert long URLs to short URLs
- **URL Redirection**: Redirect short URLs to original URLs
- **Custom URLs**: Allow users to create custom short URLs
- **Analytics**: Track click counts and usage statistics
- **User Management**: User registration and authentication
- **URL Management**: Edit, delete, and manage URLs

### Non-Functional Requirements
- **High Availability**: 99.9% uptime
- **Low Latency**: < 100ms for redirection
- **Scalability**: Handle 100M URLs per day
- **Durability**: Never lose URLs
- **Security**: Prevent abuse and malicious URLs
- **Analytics**: Real-time and historical data

### Constraints
- **Short URL Length**: 6-8 characters
- **Character Set**: Alphanumeric (a-z, A-Z, 0-9)
- **Read/Write Ratio**: 100:1 (100 reads per write)
- **URL Expiration**: Optional TTL support
- **Rate Limiting**: Prevent abuse

## 📊 Capacity Estimation

### Traffic Estimates
- **Daily URLs**: 100 million
- **URLs per second**: ~1,160
- **Read requests per second**: ~116,000
- **Write requests per second**: ~1,160
- **Peak traffic**: 3x average = ~3,480 writes/sec, ~348,000 reads/sec

### Storage Requirements
- **URL Length**: Average 100 characters
- **Short URL**: 8 characters
- **Metadata**: User ID, creation time, expiration, etc. (~200 bytes)
- **Total per URL**: ~300 bytes
- **Daily storage**: 100M × 300 bytes = 30 GB
- **Annual storage**: 30 GB × 365 = ~11 TB
- **5-year storage**: ~55 TB

### Bandwidth Requirements
- **Read requests**: 348,000/sec × 100 bytes = 34.8 MB/sec
- **Write requests**: 3,480/sec × 300 bytes = 1.04 MB/sec
- **Total bandwidth**: ~36 MB/sec

## 🏗️ High-Level Design

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │   Client    │    │   Client    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌─────────────┐
                    │ Load        │
                    │ Balancer    │
                    └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Web       │    │   Web       │    │   Web       │
│   Server    │    │   Server    │    │   Server    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌─────────────┐
                    │   Cache     │
                    │  (Redis)    │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │  Database   │
                    │ (PostgreSQL)│
                    └─────────────┘
```

## 🔧 Detailed Design

### 1. URL Shortening Algorithm

#### Base62 Encoding
```python
import string
import random

class URLShortener:
    def __init__(self):
        self.chars = string.ascii_letters + string.digits  # 62 characters
        self.base = len(self.chars)
    
    def encode(self, url_id):
        """Convert integer ID to base62 string"""
        if url_id == 0:
            return self.chars[0]
        
        result = []
        while url_id:
            result.append(self.chars[url_id % self.base])
            url_id //= self.base
        
        return ''.join(reversed(result))
    
    def decode(self, short_url):
        """Convert base62 string to integer ID"""
        url_id = 0
        for char in short_url:
            url_id = url_id * self.base + self.chars.index(char)
        return url_id
```

#### Hash-based Approach
```python
import hashlib
import base64

def shorten_url(long_url):
    # Generate MD5 hash
    hash_object = hashlib.md5(long_url.encode())
    hash_hex = hash_object.hexdigest()
    
    # Take first 6 characters
    short_code = hash_hex[:6]
    
    # Check for collisions and handle
    return short_code
```

### 2. Database Schema

#### URLs Table
```sql
CREATE TABLE urls (
    id BIGSERIAL PRIMARY KEY,
    short_code VARCHAR(8) UNIQUE NOT NULL,
    long_url TEXT NOT NULL,
    user_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    click_count BIGINT DEFAULT 0,
    INDEX idx_short_code (short_code),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);
```

#### Users Table
```sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

#### Analytics Table
```sql
CREATE TABLE url_analytics (
    id BIGSERIAL PRIMARY KEY,
    url_id BIGINT NOT NULL,
    ip_address INET,
    user_agent TEXT,
    referer TEXT,
    country VARCHAR(2),
    city VARCHAR(100),
    clicked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (url_id) REFERENCES urls(id),
    INDEX idx_url_id (url_id),
    INDEX idx_clicked_at (clicked_at)
);
```

### 3. API Design

#### REST Endpoints
```python
# Shorten URL
POST /api/v1/shorten
{
    "long_url": "https://example.com/very/long/url",
    "custom_code": "optional",
    "expires_at": "2024-12-31T23:59:59Z"
}

Response:
{
    "short_url": "https://short.ly/abc123",
    "long_url": "https://example.com/very/long/url",
    "created_at": "2024-01-01T00:00:00Z",
    "expires_at": "2024-12-31T23:59:59Z"
}

# Redirect URL
GET /{short_code}
Response: 302 Redirect to original URL

# Get URL Info
GET /api/v1/urls/{short_code}
Response:
{
    "short_url": "https://short.ly/abc123",
    "long_url": "https://example.com/very/long/url",
    "click_count": 150,
    "created_at": "2024-01-01T00:00:00Z",
    "expires_at": "2024-12-31T23:59:59Z"
}

# Get Analytics
GET /api/v1/urls/{short_code}/analytics
Response:
{
    "total_clicks": 150,
    "unique_clicks": 120,
    "clicks_by_country": {
        "US": 80,
        "CA": 40,
        "UK": 30
    },
    "clicks_by_date": {
        "2024-01-01": 50,
        "2024-01-02": 100
    }
}
```

### 4. Caching Strategy

#### Multi-Level Caching
```python
import redis
from functools import wraps

# Redis configuration
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

def cache_url(expiry=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(short_code):
            # Check cache first
            cached_url = redis_client.get(f"url:{short_code}")
            if cached_url:
                return cached_url
            
            # Get from database
            url = func(short_code)
            if url:
                redis_client.setex(f"url:{short_code}", expiry, url)
            return url
        return wrapper
    return decorator

@cache_url(expiry=3600)
def get_long_url(short_code):
    # Database query
    pass
```

#### Cache Invalidation
```python
def invalidate_cache(short_code):
    """Invalidate cache when URL is updated or deleted"""
    redis_client.delete(f"url:{short_code}")
    redis_client.delete(f"analytics:{short_code}")
```

### 5. Load Balancing

#### Round Robin Load Balancer
```nginx
upstream url_shortener {
    server web1.example.com:8000;
    server web2.example.com:8000;
    server web3.example.com:8000;
}

server {
    listen 80;
    server_name short.ly;
    
    location / {
        proxy_pass http://url_shortener;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Health Checks
```python
@app.route('/health')
def health_check():
    # Check database connection
    db_status = check_database_connection()
    
    # Check cache connection
    cache_status = check_cache_connection()
    
    if db_status and cache_status:
        return {'status': 'healthy'}, 200
    else:
        return {'status': 'unhealthy'}, 503
```

## 🚀 Scalability Considerations

### 1. Database Sharding

#### Horizontal Sharding by URL ID
```python
class DatabaseShard:
    def __init__(self, shard_id, connection_string):
        self.shard_id = shard_id
        self.connection = create_connection(connection_string)
    
    def get_shard_for_url_id(self, url_id):
        return url_id % self.num_shards

# Shard configuration
SHARDS = {
    0: "postgresql://shard0.example.com/urls",
    1: "postgresql://shard1.example.com/urls",
    2: "postgresql://shard2.example.com/urls",
    3: "postgresql://shard3.example.com/urls"
}
```

### 2. Read Replicas

#### Master-Slave Replication
```python
class DatabaseCluster:
    def __init__(self):
        self.master = create_connection(MASTER_DB_URL)
        self.slaves = [
            create_connection(SLAVE1_DB_URL),
            create_connection(SLAVE2_DB_URL),
            create_connection(SLAVE3_DB_URL)
        ]
        self.slave_index = 0
    
    def get_read_connection(self):
        # Round-robin read replicas
        connection = self.slaves[self.slave_index]
        self.slave_index = (self.slave_index + 1) % len(self.slaves)
        return connection
    
    def get_write_connection(self):
        return self.master
```

### 3. CDN Integration

#### CloudFlare Configuration
```javascript
// CloudFlare Workers for edge caching
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  const shortCode = url.pathname.slice(1)
  
  // Check edge cache first
  let response = await caches.default.match(request)
  if (response) {
    return response
  }
  
  // Fetch from origin
  response = await fetch(`https://api.short.ly/${shortCode}`)
  
  // Cache for 1 hour
  const cacheResponse = response.clone()
  cacheResponse.headers.set('Cache-Control', 'max-age=3600')
  event.waitUntil(caches.default.put(request, cacheResponse))
  
  return response
}
```

## 🔒 Security Considerations

### 1. Rate Limiting

#### Token Bucket Algorithm
```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = defaultdict(lambda: capacity)
        self.last_refill = defaultdict(time.time)
    
    def is_allowed(self, key):
        now = time.time()
        time_passed = now - self.last_refill[key]
        
        # Refill tokens
        self.tokens[key] = min(
            self.capacity,
            self.tokens[key] + time_passed * self.refill_rate
        )
        self.last_refill[key] = now
        
        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True
        return False

# Rate limiting: 100 requests per minute per IP
rate_limiter = RateLimiter(capacity=100, refill_rate=100/60)
```

### 2. URL Validation

#### Malicious URL Detection
```python
import re
from urllib.parse import urlparse

class URLValidator:
    def __init__(self):
        self.malicious_patterns = [
            r'phishing',
            r'malware',
            r'spam',
            # Add more patterns
        ]
        self.allowed_domains = [
            'example.com',
            'google.com',
            # Add trusted domains
        ]
    
    def is_valid_url(self, url):
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Check for malicious patterns
            for pattern in self.malicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
            
            # Check domain whitelist (if applicable)
            if self.allowed_domains and parsed.netloc not in self.allowed_domains:
                return False
            
            return True
        except:
            return False
```

### 3. Authentication & Authorization

#### JWT Token Authentication
```python
import jwt
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
    
    def generate_token(self, user_id):
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
```

## 📊 Monitoring & Analytics

### 1. Metrics Collection

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
url_shortens_total = Counter('url_shortens_total', 'Total URL shortenings')
url_redirects_total = Counter('url_redirects_total', 'Total URL redirects')
url_redirect_duration = Histogram('url_redirect_duration_seconds', 'URL redirect duration')
active_urls = Gauge('active_urls_total', 'Total active URLs')

@app.route('/shorten', methods=['POST'])
def shorten_url():
    # ... shortening logic ...
    url_shortens_total.inc()
    active_urls.inc()
    return response

@app.route('/<short_code>')
def redirect_url():
    start_time = time.time()
    # ... redirection logic ...
    url_redirects_total.inc()
    url_redirect_duration.observe(time.time() - start_time)
    return response
```

### 2. Logging

#### Structured Logging
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger('url_shortener')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_url_shorten(self, short_code, long_url, user_id):
        log_data = {
            'event': 'url_shorten',
            'short_code': short_code,
            'long_url': long_url,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_url_redirect(self, short_code, ip_address, user_agent):
        log_data = {
            'event': 'url_redirect',
            'short_code': short_code,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
```

## 🚀 Deployment & DevOps

### 1. Docker Configuration

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/urls
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=urls
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 2. Kubernetes Deployment

#### Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: url-shortener
spec:
  replicas: 3
  selector:
    matchLabels:
      app: url-shortener
  template:
    metadata:
      labels:
        app: url-shortener
    spec:
      containers:
      - name: url-shortener
        image: url-shortener:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 📈 Performance Optimization

### 1. Database Optimization

#### Indexing Strategy
```sql
-- Primary indexes
CREATE INDEX idx_urls_short_code ON urls(short_code);
CREATE INDEX idx_urls_user_id ON urls(user_id);
CREATE INDEX idx_urls_created_at ON urls(created_at);

-- Composite indexes
CREATE INDEX idx_urls_user_created ON urls(user_id, created_at);
CREATE INDEX idx_analytics_url_clicked ON url_analytics(url_id, clicked_at);

-- Partial indexes
CREATE INDEX idx_urls_active ON urls(short_code) WHERE is_active = TRUE;
```

#### Query Optimization
```python
# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# Use prepared statements
def get_url_by_short_code(short_code):
    query = text("SELECT long_url FROM urls WHERE short_code = :code AND is_active = TRUE")
    result = engine.execute(query, code=short_code)
    return result.fetchone()
```

### 2. Caching Optimization

#### Cache Warming
```python
def warm_cache():
    """Pre-load popular URLs into cache"""
    popular_urls = get_popular_urls(limit=10000)
    
    for url in popular_urls:
        cache_key = f"url:{url.short_code}"
        redis_client.setex(cache_key, 3600, url.long_url)
```

#### Cache Compression
```python
import gzip
import json

def compress_data(data):
    return gzip.compress(json.dumps(data).encode())

def decompress_data(compressed_data):
    return json.loads(gzip.decompress(compressed_data))
```

## 🎯 Conclusion

This URL shortener design provides:

1. **High Scalability**: Handles 100M URLs per day
2. **Low Latency**: < 100ms redirection time
3. **High Availability**: 99.9% uptime
4. **Security**: Rate limiting, URL validation, authentication
5. **Analytics**: Real-time and historical data
6. **Monitoring**: Comprehensive metrics and logging

The system can be further optimized with:
- Machine learning for URL classification
- Advanced caching strategies
- Global CDN deployment
- Real-time analytics processing
- A/B testing for optimization

---

**Happy Designing! 🚀**
