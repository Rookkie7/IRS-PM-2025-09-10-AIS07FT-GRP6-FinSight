from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

_client = AsyncIOMotorClient(settings.MONGO_URI)
db = _client[settings.MONGO_DB]