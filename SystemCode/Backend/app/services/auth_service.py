from fastapi import HTTPException, status, Depends
from jose import JWTError, jwt

from app.config import settings
from app.ports.storage import UserRepoPort
from app.model.models import UserCreate, UserInDB
from app.utils.security import hash_password, verify_password, create_access_token

class AuthService:
    def __init__(self, repo: UserRepoPort):
        self.repo = repo

    async def register(self, payload: UserCreate) -> str:
        uid = await self.repo.create_user(payload, hash_password(payload.password))
        return uid

    async def authenticate(self, email: str, password: str) -> str:
        user: UserInDB | None = await self.repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
        return create_access_token(subject=user.id)

    async def get_current_user(self, token: str) -> UserInDB:
        credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
        try:
            payload = jwt.decode(token, settings.AUTH_SECRET_KEY, algorithms=[settings.AUTH_ALGORITHM])
            uid: str | None = payload.get("sub")
            if uid is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
        user = await self.repo.get_by_id(uid)
        if not user or not user.is_active:
            raise credentials_exception
        return user