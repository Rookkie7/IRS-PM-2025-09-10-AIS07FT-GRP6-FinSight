from fastapi import APIRouter, Depends
from app.services.rec_service import RecService
from app.deps import get_rec_service

router = APIRouter(prefix="/rec", tags=["recommendation"])

# @router.post("/user")
# async def rec_user(payload: dict, svc: RecService = Depends(get_rec_service)):
#    ...