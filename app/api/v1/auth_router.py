from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr

from app.adapters.db.user_repo_mongo import UserRepoMongo
from app.model.models import UserCreate, Token, UserPublic, UserInDB
from app.ports.storage import UserRepoPort
from app.services.auth_service import AuthService
from app.deps import get_auth_service, get_embedder, get_user_repo

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
@router.post("/register", response_model=UserPublic)
async def register_user(payload: UserCreate,
                        svc: AuthService = Depends(get_auth_service),
                        embedder=Depends(get_embedder),
                        user_repo: UserRepoMongo = Depends(get_user_repo)):
    uid = await svc.register(payload, embedder=embedder, dim=32)
    u: UserInDB | None = await user_repo.get_by_id(uid)
    assert u is not None, "user not found"
    return to_user_public(u)

@router.post("/login", response_model=Token)
async def login(payload: LoginIn, svc: AuthService = Depends(get_auth_service)):
    token = await svc.authenticate(payload.email, payload.password)
    return Token(access_token=token)