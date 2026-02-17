# 🚀 Deployment Guide: FlyWire Simulator on EC2 + Vercel

This guide covers deploying the FlyWire Connectome Simulator with:
- **EC2 GPU Instance**: Backend (FastAPI + CUDA + PyTorch)
- **Vercel**: Frontend (Static HTML/JS)

---

## 📋 Prerequisites

### AWS EC2
- AWS account with EC2 access
- Recommended instance: **g4dn.xlarge** or similar GPU instance
- Ubuntu 22.04 LTS AMI
- Security group allowing:
  - Port 22 (SSH)
  - Port 8000 (API)
  - Port 443 (HTTPS, if using SSL)

### Vercel
- Vercel account (free tier works)
- Vercel CLI installed: `npm install -g vercel`

---

## 🖥️ Part 1: EC2 Backend Deployment

### Step 1: Launch EC2 Instance

1. **Choose AMI**: Ubuntu 22.04 LTS
2. **Instance Type**: g4dn.xlarge (1 GPU, 4 vCPUs, 16 GB RAM)
   - Alternative: g4dn.2xlarge for better performance
3. **Storage**: 50 GB minimum (100 GB recommended)
4. **Security Group**: 
   ```
   SSH (22): Your IP
   Custom TCP (8000): 0.0.0.0/0
   HTTPS (443): 0.0.0.0/0 (optional)
   ```

### Step 2: Connect and Setup

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Upload the setup script
scp -i your-key.pem ec2-setup.sh ubuntu@your-ec2-ip:~/

# Run setup script
chmod +x ec2-setup.sh
./ec2-setup.sh

# This installs:
# - Docker
# - NVIDIA Container Toolkit
# - Docker Compose
```

### Step 3: Upload Application Code

```bash
# From your local machine
scp -i your-key.pem -r FastFly ubuntu@your-ec2-ip:~/

# Or use git
ssh -i your-key.pem ubuntu@your-ec2-ip
git clone <your-repo-url> FastFly
cd FastFly
```

### Step 4: Configure Environment

```bash
# Create .env file from example
cp .env.example .env

# Edit with your Vercel URL
nano .env
```

Update `.env`:
```bash
ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:3000
HOST=0.0.0.0
PORT=8000
```

### Step 5: Build and Run

```bash
# Test GPU access first
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi

# Build and start the application
docker compose up -d

# Check logs
docker compose logs -f

# Verify it's running
curl http://localhost:8000/api/health
```

### Step 6: Test the API

```bash
# From your local machine
curl http://your-ec2-ip:8000/api/health

# Expected response:
# {"status":"healthy","n_neurons":139255,"n_synapses":54500000}
```

---

## 🌐 Part 2: Vercel Frontend Deployment

### Step 1: Update Frontend for Cross-Origin API Calls

The frontend needs to know where the backend API is. You have two options:

#### Option A: Update HTML files manually

Add this script tag at the beginning of `<body>` in each HTML file (`index.html`, `io_analysis.html`, `io_analysis_torch.html`):

```html
<script>
    // API Configuration
    const API_BASE_URL = 'http://your-ec2-ip:8000';  // Replace with your EC2 URL
    const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');
    
    window.getApiUrl = (path) => API_BASE_URL + path;
    window.getWsUrl = (path) => WS_BASE_URL + path;
    
    // Override fetch for API calls
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        if (typeof url === 'string' && url.startsWith('/api/')) {
            url = API_BASE_URL + url;
        }
        return originalFetch(url, options);
    };
</script>
```

Then update WebSocket connections in `index.html` around line 1558:
```javascript
// Before:
ws = new WebSocket(`${proto}//${location.host}/ws`);

// After:
ws = new WebSocket(window.getWsUrl('/ws'));
```

#### Option B: Use environment variable (recommended)

1. Include `config.js` in your HTML files:
```html
<script src="/static/config.js"></script>
```

2. Configure in Vercel dashboard (see Step 3 below)

### Step 2: Deploy to Vercel

```bash
# From your FastFly project directory
vercel login

# Deploy (first time)
vercel

# Follow prompts:
# - Setup new project? Yes
# - Which directory? ./
# - Override settings? No

# The output directory is 'static' (configured in vercel.json)
```

### Step 3: Configure Environment Variables in Vercel

1. Go to https://vercel.com/dashboard
2. Select your project
3. Go to **Settings** → **Environment Variables**
4. Add:
   ```
   VITE_API_URL = http://your-ec2-ip:8000
   ```
   Or if using HTTPS:
   ```
   VITE_API_URL = https://api.yourdomain.com
   ```

### Step 4: Deploy Production

```bash
# Deploy to production
vercel --prod
```

Your app will be live at: `https://your-app.vercel.app`

---

## 🔒 Part 3: Security & Production Hardening

### 1. Setup HTTPS for EC2 (Recommended)

#### Option A: Use Nginx as reverse proxy with Let's Encrypt

```bash
# Install Nginx and Certbot
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/fastfly
```

Nginx config:
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/fastfly /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d api.yourdomain.com
```

#### Option B: Use Application Load Balancer

1. Create ALB in AWS Console
2. Add SSL certificate from ACM
3. Target your EC2 instance on port 8000

### 2. Update CORS Origins

After setting up HTTPS, update EC2 `.env`:
```bash
ALLOWED_ORIGINS=https://your-app.vercel.app
```

Restart:
```bash
docker compose down
docker compose up -d
```

### 3. Secure EC2 Security Group

Update to allow only necessary traffic:
```
SSH (22): Your IP only
HTTP (80): 0.0.0.0/0 (if using Nginx)
HTTPS (443): 0.0.0.0/0
Custom TCP (8000): Remove public access if using Nginx/ALB
```

---

## 🧪 Testing

### Test Backend
```bash
# Health check
curl https://api.yourdomain.com/api/health

# Get neuron positions
curl https://api.yourdomain.com/api/positions
```

### Test Frontend
1. Visit `https://your-app.vercel.app`
2. Open browser console (F12)
3. Check for API configuration log
4. Try connecting to simulator

### Test WebSocket
```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c ws://your-ec2-ip:8000/ws

# Or with WSS
wscat -c wss://api.yourdomain.com/ws
```

---

## 📊 Monitoring & Maintenance

### View Logs
```bash
# Real-time logs
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100
```

### Restart Service
```bash
docker compose restart
```

### Update Application
```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose down
docker compose up -d --build
```

### Monitor GPU Usage
```bash
# Inside EC2
nvidia-smi

# Or continuously
watch -n 1 nvidia-smi
```

---

## 💰 Cost Estimates

### EC2 (g4dn.xlarge)
- **On-Demand**: ~$0.526/hour (~$378/month)
- **Spot Instance**: ~$0.158/hour (~$113/month)
- **Reserved (1-year)**: ~$0.316/hour (~$227/month)

### Vercel
- **Free Tier**: Usually sufficient
- **Pro**: $20/month (if needed for extra bandwidth)

### Optimization Tips
1. Use **Spot Instances** for development
2. **Stop EC2** when not in use (saves ~90%)
3. Use **Reserved Instances** for production
4. Consider **SageMaker** for auto-scaling

---

## 🐛 Troubleshooting

### Backend Issues

**Problem**: Container fails to start
```bash
# Check logs
docker compose logs

# Check GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi

# Rebuild
docker compose down
docker compose build --no-cache
docker compose up -d
```

**Problem**: CUDA errors
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall NVIDIA container toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

**Problem**: CORS errors
- Update `ALLOWED_ORIGINS` in `.env`
- Restart container: `docker compose restart`

### Frontend Issues

**Problem**: Cannot connect to API
- Check browser console for errors
- Verify `VITE_API_URL` in Vercel dashboard
- Check EC2 security group allows port 8000
- Test API directly: `curl http://your-ec2-ip:8000/api/health`

**Problem**: WebSocket connection fails
- Check WebSocket URL in browser console
- Verify EC2 allows WebSocket upgrades
- If using Nginx, ensure proxy_set_header for Upgrade

---

## 🔄 CI/CD (Optional)

### GitHub Actions for EC2

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to EC2

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to EC2
        env:
          PRIVATE_KEY: ${{ secrets.EC2_SSH_KEY }}
          HOST: ${{ secrets.EC2_HOST }}
          USER: ubuntu
        run: |
          echo "$PRIVATE_KEY" > private_key
          chmod 600 private_key
          ssh -o StrictHostKeyChecking=no -i private_key ${USER}@${HOST} '
            cd ~/FastFly &&
            git pull &&
            docker compose down &&
            docker compose up -d --build
          '
```

### Vercel Auto-Deploy
- Vercel automatically deploys on git push
- Configure in Vercel dashboard → Git Integration

---

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Vercel Documentation](https://vercel.com/docs)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/g4/)

---

## ✅ Checklist

Backend Setup:
- [ ] EC2 instance launched with GPU
- [ ] Docker and NVIDIA toolkit installed
- [ ] Application code uploaded
- [ ] Environment variables configured
- [ ] Container running successfully
- [ ] Health check passes
- [ ] CORS configured

Frontend Setup:
- [ ] HTML files updated with API URL
- [ ] Deployed to Vercel
- [ ] Environment variables set
- [ ] Can access frontend
- [ ] API calls working
- [ ] WebSocket connection working

Security:
- [ ] HTTPS configured (recommended)
- [ ] Security groups locked down
- [ ] CORS origins restricted
- [ ] Monitoring setup

---

## 🆘 Support

If you encounter issues:

1. Check logs: `docker compose logs -f`
2. Verify GPU: `nvidia-smi`
3. Test API: `curl http://localhost:8000/api/health`
4. Check network: Security groups, CORS, firewalls
5. Review browser console for frontend errors

For questions, open an issue in the repository.

---

**Good luck with your deployment! 🚀**
