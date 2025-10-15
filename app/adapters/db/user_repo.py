from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.adapters.db.database_client import get_mongo_db, get_postgres_session
from app.model.exception import UserAlreadyExistsError
from app.model.models import UserInDB, EmbeddingVector, UserCreate
from app.ports.storage import UserRepoPort


class UserRepo(UserRepoPort):
    def __init__(self):
        self.col = get_mongo_db()["users"]   # 这里只拿句柄，不做 IO

    async def ensure_indexes(self) -> None:
        # 这里才是异步 IO，必须 await，在有事件循环的上下文里调用一次即可
        await self.col.create_index("email", unique=True)
        await self.col.create_index("username", unique=True)

    @staticmethod
    def _to_user(doc) -> UserInDB:
        if not doc:
            return None
        return UserInDB(
            id=str(doc["_id"]),
            email=doc["email"],
            username=doc["username"],
            hashed_password=doc["hashed_password"],
            created_at=doc["created_at"],
            updated_at=doc.get("updated_at", doc["created_at"]),
            profile=doc.get("profile", {}),
            embedding=EmbeddingVector(**doc["embedding"]) if doc.get("embedding") else None,
            is_active=doc.get("is_active", True),
        )

    async def create_user(self, user: UserCreate, hashed_password: str) -> str:
        doc = {
            "email": user.email,
            "username": user.username,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "profile": {
                "full_name": user.full_name,
                "bio": user.bio,
                "interests": user.interests,
                "sectors": user.sectors,
                "tickers": user.tickers,
            },
            "is_active": True,
        }
        try:
            res = await self.col.insert_one(doc)
            return str(res.inserted_id)
        except DuplicateKeyError as e:
            # 判断是 email 还是 username 冲突
            msg = str(e)
            if "email_" in msg or " dup key: { email:" in msg:
                raise UserAlreadyExistsError("email", user.email)
            if "username_" in msg or " dup key: { username:" in msg:
                raise UserAlreadyExistsError("username", user.username)
            raise

    async def get_by_email(self, email: str) -> Optional[UserInDB]:
        doc = await self.col.find_one({"email": email})
        return self._to_user(doc)

    async def get_by_id(self, uid: str) -> Optional[UserInDB]:
        try:
            oid = ObjectId(uid)
        except Exception:
            return None
        doc = await self.col.find_one({"_id": oid})
        return self._to_user(doc)

    async def update_profile(self, uid: str, profile: dict) -> None:
        try:
            oid = ObjectId(uid)
        except Exception:
            return
        update = {
            "$set": {
                "profile": profile,
                "updated_at": datetime.utcnow(),
            }
        }
        await self.col.update_one({"_id": oid}, update)