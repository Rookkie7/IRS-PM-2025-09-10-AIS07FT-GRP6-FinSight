from typing import Optional, List
from app.ports.storage import UserRepoPort
from app.model.models import UserInDB

class UserService:
    def __init__(self, repo: UserRepoPort, embedder, dim: int = 32):
        self.repo = repo
        self.embedder = embedder
        self.dim = dim

    async def update_profile_and_embed(self, user: UserInDB, profile: dict) -> None:
        text = "\n".join([f"{k}: {v}" for k, v in profile.items() if v])
        vec = (await self.embedder.embed([text or user.username], dim=self.dim))[0]
        await self.repo.update_profile_and_embedding(user.id, profile, vec)