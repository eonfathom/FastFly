# 🚀 FastFly Deployment - Files Created

This document lists all deployment-related files created for EC2 + Vercel deployment.

## 📁 Files Overview

### Docker & EC2 Configuration
- **Dockerfile** - Production container image with CUDA support
- **docker-compose.yml** - Container orchestration with GPU support
- **.dockerignore** - Files to exclude from Docker build
- **ec2-setup.sh** - Initial EC2 instance setup script
- **deploy-ec2.sh** - Quick deployment script for updates
- **.env.example** - Environment variables template for backend

### Vercel Configuration
- **vercel.json** - Vercel deployment configuration
- **.env.vercel.example** - Environment variables for Vercel
- **static/config.js** - Frontend API configuration helper
- **configure_frontend.py** - Script to update HTML files for deployment

### Documentation
- **DEPLOYMENT.md** - Complete deployment guide (25+ pages)
- **DEPLOY-QUICKREF.md** - Quick reference for common tasks
- **nginx-config.md** - Nginx reverse proxy configuration (optional)

### Testing
- **start-local.sh** - Local development server script

### Modified Files
- **app_server.py** - Added CORS middleware and health check endpoint

## 🎯 Quick Start Guide

### 1. Backend Deployment (EC2)

```bash
# On your local machine
scp -i your-key.pem -r FastFly ubuntu@your-ec2-ip:~/
ssh -i your-key.pem ubuntu@your-ec2-ip

# On EC2 instance
cd FastFly
chmod +x ec2-setup.sh deploy-ec2.sh
./ec2-setup.sh  # First time only

# Configure environment
cp .env.example .env
nano .env  # Update ALLOWED_ORIGINS with your Vercel URL

# Deploy
./deploy-ec2.sh
```

### 2. Frontend Deployment (Vercel)

**Option A: Automatic (recommended)**
```bash
# Push to git, Vercel auto-deploys
git add .
git commit -m "Deploy to production"
git push
```

**Option B: Manual**
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel login
vercel --prod
```

**Option C: Configure HTML files manually**
```bash
# Update HTML files to point to your EC2 API
python configure_frontend.py --api-url https://your-ec2-url:8000

# Then deploy to Vercel
vercel --prod
```

### 3. Environment Configuration

**EC2 (.env file):**
```bash
ALLOWED_ORIGINS=https://your-app.vercel.app
HOST=0.0.0.0
PORT=8000
```

**Vercel (Dashboard → Settings → Environment Variables):**
```
VITE_API_URL=https://your-ec2-url:8000
```

## 🔍 Verification Checklist

After deployment, verify:

- [ ] Backend health: `curl https://your-api-url/api/health`
- [ ] Frontend loads: Visit your Vercel URL
- [ ] Browser console shows correct API config
- [ ] Can connect to simulator
- [ ] WebSocket connection works
- [ ] GPU is being used: `nvidia-smi` on EC2

## 📊 Architecture

```
┌─────────────────┐
│   Browser       │
│  (User)         │
└────────┬────────┘
         │
         │ HTTPS
         ▼
┌─────────────────┐
│   Vercel        │
│  (Frontend)     │
│  Static Files   │
└────────┬────────┘
         │
         │ API Calls / WebSocket
         │ (CORS enabled)
         ▼
┌─────────────────┐     ┌──────────────┐
│   EC2 Instance  │────▶│  NVIDIA GPU  │
│  (Backend)      │     │  (CUDA)      │
│  FastAPI        │     └──────────────┘
│  + PyTorch      │
│  + CuPy         │
└─────────────────┘
```

## 💡 Tips

1. **Use Spot Instances** for EC2 to save 70% on costs
2. **Stop EC2** when not in use (saves ~90%)
3. **Enable CloudWatch** for monitoring
4. **Set up alerts** for high GPU usage
5. **Use HTTPS** in production (see nginx-config.md)
6. **Restrict CORS** origins in production
7. **Regular backups** of neuron_annotations.npz

## 🐛 Common Issues

**CORS errors?**
- Update `ALLOWED_ORIGINS` in EC2 `.env`
- Restart: `docker compose restart`

**Can't connect to API?**
- Check EC2 security group allows port 8000
- Verify container is running: `docker ps`
- Check logs: `docker compose logs -f`

**GPU not detected?**
- Run: `docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi`
- Reinstall nvidia-container-toolkit

**High costs?**
- Stop EC2 when not needed
- Use Spot instances
- Set up auto-shutdown schedule

## 📚 Next Steps

1. Set up monitoring (CloudWatch, Datadog, etc.)
2. Configure auto-scaling if needed
3. Set up CI/CD pipeline
4. Add authentication if required
5. Configure custom domain
6. Set up SSL certificates
7. Create backup strategy

For detailed instructions, see **DEPLOYMENT.md**

## 🆘 Support

- Full deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Quick reference: [DEPLOY-QUICKREF.md](DEPLOY-QUICKREF.md)
- Nginx config: [nginx-config.md](nginx-config.md)

---

Created: 2026-02-13
Version: 1.0
