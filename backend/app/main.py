from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.api.ping import router as ping_router
from backend.app.api.info import router as info_router
from backend.app.api.time_api import router as time_router
from backend.app.api.greet import router as greet_router
from backend.app.api.add import router as add_router
from backend.app.api.upload import router as upload_router
from backend.app.api.documents import router as documents_router
from backend.app.api.query import router as query_router

app = FastAPI(title="OmniBrain Backend")

app.include_router(health_router)
app.include_router(ping_router)
app.include_router(info_router)
app.include_router(time_router)
app.include_router(greet_router)
app.include_router(add_router)
app.include_router(upload_router)
app.include_router(documents_router)
app.include_router(documents_router)
app.include_router(query_router)

@app.get("/")
def root():
    return {"message": "OmniBrain Backend Running"}