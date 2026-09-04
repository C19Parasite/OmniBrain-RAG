import argparse, uvicorn
from config.settings import HOST, PORT

def main():
    parser = argparse.ArgumentParser(description="OmniBrain Studio CLI")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    print(f"Launching OmniBrain Studio on http://{args.host}:{args.port}")
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
