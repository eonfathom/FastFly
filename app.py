import uvicorn
from app_server import app, args

if __name__ == "__main__":
    print(f"Starting Fast Fly server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
