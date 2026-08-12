from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/time")
def get_time():
    return {"server_time": datetime.now().isoformat()}