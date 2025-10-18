from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr

from app.adapters.db.user_repo import UserRepo
from app.model.models import UserCreate, Token, UserPublic, UserInDB, RegisterResponse
from app.services.auth_service import AuthService
from app.deps import get_auth_service, get_user_repo, get_user_service
from app.services.stock_service import SECTOR_LIST
from app.services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginIn(BaseModel):
    email: EmailStr
    password: str

def to_user_public(u: UserInDB) -> UserPublic:
    return UserPublic(
        id=u.id,
        email=u.email,
        username=u.username,
        created_at=u.created_at,
        profile=u.profile,
        embedding=u.embedding,
    )

# 将用户信息存入mongodb，将用户的向量存入pgvector
@router.post("/register", response_model=UserPublic)
async def register_user(payload: UserCreate,
                        profile_data: Dict[str, Any] = None,
                        auth_svc: AuthService = Depends(get_auth_service),
                        user_svc: UserService = Depends(get_user_service),
                        user_repo: UserRepo = Depends(get_user_repo)):
    uid = await auth_svc.register(payload)
    if profile_data is None:
        profile_data = {}
    selected = set(payload.sectors or [])
    industry_vector = [
        0.5 if sector in selected else 0.0
        for sector in SECTOR_LIST
    ]
    profile_data["industry_preferences"] = industry_vector
    profile_data["investment_preferences"] = list(payload.investment_preference.values())
    print(profile_data)

    user_svc.init_user_profile(uid,False, profile_data)
    u: UserInDB | None = await user_repo.get_by_id(uid)
    assert u is not None, "user not found"
    return to_user_public(u)

@router.post("/login", response_model=Token)
async def login(payload: LoginIn, svc: AuthService = Depends(get_auth_service)):
    token = await svc.authenticate(payload.email, payload.password)
    return Token(access_token=token)