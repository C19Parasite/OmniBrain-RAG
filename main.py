"""
OmniBrain Studio ? Root Application Entrypoint.
"""
import uvicorn
from config.settings import HOST, PORT

if __name__ == "__main__":
    print(f"===================================================")
    print(f"  Starting OmniBrain Studio on http://{HOST}:{PORT}")
    print(f"===================================================")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
