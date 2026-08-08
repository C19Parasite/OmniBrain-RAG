from fastapi import FastAPI

app = FastAPI(title="OmniBrain Backend")

@app.get("/")
def root():
    return {"message": "OmniBrain Backend Running"}

@app.get("/health")
def health():
    return {"status": "ok"}