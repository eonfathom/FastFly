# Deployment Quick Reference

## Local Development
```bash
python app.py --host 127.0.0.1 --port 8000
# Visit http://localhost:8000
```

## Production Deployment

### 1. Backend (EC2)
```bash
# On EC2 instance
cd ~/FastFly
docker compose up -d
```

### 2. Frontend (Vercel)
```bash
# From local machine
vercel --prod
```

### 3. Update after code changes

**Backend:**
```bash
ssh ubuntu@your-ec2-ip
cd ~/FastFly
git pull
docker compose down
docker compose up -d --build
```

**Frontend:**
```bash
git push  # Vercel auto-deploys
# Or manually: vercel --prod
```

## Configuration Files

- **EC2 Backend**: `.env` (copy from `.env.example`)
- **Vercel Frontend**: Set `VITE_API_URL` in Vercel dashboard
- **CORS**: Update `ALLOWED_ORIGINS` in backend `.env`

## Monitoring

**Backend logs:**
```bash
docker compose logs -f
```

**GPU usage:**
```bash
nvidia-smi
```

**Health check:**
```bash
curl https://your-api-url/api/health
```

## Costs

- **EC2 g4dn.xlarge**: ~$0.526/hr ($378/mo on-demand)
- **EC2 Spot**: ~$0.158/hr ($113/mo)
- **Vercel**: Free tier (Pro $20/mo if needed)

**Tip**: Stop EC2 when not in use to save ~90% on costs!

For complete guide, see [DEPLOYMENT.md](DEPLOYMENT.md)
