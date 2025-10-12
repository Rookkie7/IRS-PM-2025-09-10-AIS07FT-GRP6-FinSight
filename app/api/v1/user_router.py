from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.model.models import UserPublic
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.deps import get_auth_service, get_user_service

router = APIRouter(prefix="/users", tags=["users"])

class ProfileUpdateIn(BaseModel):
    full_name: str | None = None
    bio: str | None = None
    interests: list[str] = []
    sectors: list[str] = []
    tickers: list[str] = []

@router.get("/me", response_model=UserPublic)
async def me(auth: AuthService = Depends(get_auth_service)):
    u = await auth.get_current_user()
    return UserPublic(
        id=u.id, email=u.email, username=u.username,
        created_at=u.created_at, profile=u.profile, embedding=u.embedding
    )

@router.put("/me")
async def update_me(payload: ProfileUpdateIn, auth: AuthService = Depends(get_auth_service), usvc: UserService = Depends(get_user_service)):
    u = await auth.get_current_user()
    profile = {
        "full_name": payload.full_name,
        "bio": payload.bio,
        "interests": payload.interests,
        "sectors": payload.sectors,
        "tickers": payload.tickers,
    }
    await usvc.update_profile_and_embed(u, profile)
    return {"ok": True}