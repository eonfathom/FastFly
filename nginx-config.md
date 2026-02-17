nginx.conf (optional reverse proxy):
```nginx
# /etc/nginx/sites-available/fastfly

server {
    listen 80;
    server_name api.yourdomain.com;  # Replace with your domain

    # Redirect HTTP to HTTPS (after SSL is configured)
    # return 301 https://$server_name$request_uri;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        
        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Standard proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

# HTTPS configuration (uncomment after running certbot)
# server {
#     listen 443 ssl http2;
#     server_name api.yourdomain.com;
# 
#     ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
#     ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
#     include /etc/letsencrypt/options-ssl-nginx.conf;
#     ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
# 
#     location / {
#         proxy_pass http://localhost:8000;
#         proxy_http_version 1.1;
#         
#         proxy_set_header Upgrade $http_upgrade;
#         proxy_set_header Connection "upgrade";
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto $scheme;
#         
#         proxy_connect_timeout 60s;
#         proxy_send_timeout 60s;
#         proxy_read_timeout 60s;
#     }
# }
```

## Nginx Setup Commands

```bash
# Install Nginx and Certbot
sudo apt-get update
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Create config
sudo nano /etc/nginx/sites-available/fastfly

# Enable site
sudo ln -s /etc/nginx/sites-available/fastfly /etc/nginx/sites-enabled/

# Test config
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Get SSL certificate (make sure DNS is pointing to your EC2)
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal is configured automatically by certbot
# Test renewal with:
sudo certbot renew --dry-run
```

## systemd service (alternative to docker-compose)

If you prefer to run without Docker:

```ini
# /etc/systemd/system/fastfly.service

[Unit]
Description=FastFly Connectome Simulator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/FastFly
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/usr/bin/python3 /home/ubuntu/FastFly/app.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fastfly
sudo systemctl start fastfly
sudo systemctl status fastfly

# View logs
sudo journalctl -u fastfly -f
```
