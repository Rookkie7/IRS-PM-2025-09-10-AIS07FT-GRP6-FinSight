import numpy as np
import logging
from typing import List, Dict, Any, Optional
import random
from fastapi import Depends
from datetime import datetime

from sqlalchemy.orm.session import Session

from app.adapters.db.database_client import get_postgres_session
from app.model.models import UserProfile

logger = logging.getLogger(__name__)

# 行业列表
SECTOR_LIST = [
    'Utilities', 'Technology', 'Consumer Defensive', 'Healthcare',
    'Basic Materials', 'Real Estate', 'Energy', 'Industrials',
    'Consumer Cyclical', 'Communication Services', 'Financial Services'
]


from typing import Optional, List
from app.ports.storage import UserRepoPort
from app.model.models import UserInDB

class UserService:
    def __init__(self, db: Session,repo: UserRepoPort, embedder, dim: int = 20):
        self.db = db
        self.sector_list = SECTOR_LIST
        self.repo = repo
        self.embedder = embedder
        self.dim = dim

    def init_user_profile(self, user_id: str, reset: bool = False,
                          profile_data: Dict[str, Any] = None) -> Optional[UserProfile]:
        """
        初始化用户画像（20维向量）
        """
        try:
            existing = self.get_user_profile(user_id)

            if existing and not reset:
                logger.info(f"用户 {user_id} 画像已存在")
                return existing

            # 使用提供的画像数据或创建默认画像
            if profile_data:
                user_profile = self._create_profile_from_data(user_id, profile_data)
            else:
                user_profile = self._create_default_profile(user_id)

            if existing and reset:
                # 更新现有画像
                self._update_existing_profile(existing, user_profile)
                logger.info(f"用户 {user_id} 画像已重置")
            else:
                # 创建新画像
                self.db.add(user_profile)
                logger.info(f"用户 {user_id} 画像创建成功")

            self.db.commit()
            return self.get_user_profile(user_id)

        except Exception as e:
            logger.error(f"初始化用户 {user_id} 画像失败: {str(e)}")
            self.db.rollback()
            return None

    def _create_default_profile(self, user_id: str) -> UserProfile:
        """创建默认用户画像"""
        profile = UserProfile(user_id=user_id)

        # 1. 行业偏好 (11维) - 均匀分布
        industry_prefs = [0.5] * 11  # 中性偏好

        # 2. 投资偏好 (9维) - 中性偏好
        investment_prefs = [0.5] * 9

        profile.industry_preferences = industry_prefs
        profile.investment_preferences = investment_prefs

        # 构建20维向量
        vector_20d = profile.build_vector_from_components()
        profile.set_profile_vector_20d(vector_20d)

        return profile

    def _create_profile_from_data(self, user_id: str, profile_data: Dict[str, Any]) -> UserProfile:
        """从数据创建用户画像"""
        profile = UserProfile(user_id=user_id)

        # 设置各个维度
        profile.industry_preferences = profile_data.get('industry_preferences', [0.5] * 11)
        profile.investment_preferences = profile_data.get('investment_preferences', [0.5] * 9)

        # 构建20维向量
        vector_20d = profile.build_vector_from_components()
        profile.set_profile_vector_20d(vector_20d)


        return profile

    def _update_existing_profile(self, existing: UserProfile, new_data: UserProfile):
        """更新现有用户画像"""
        existing.industry_preferences = new_data.industry_preferences
        existing.investment_preferences = new_data.investment_preferences
        existing.set_profile_vector_20d(new_data.get_profile_vector_20d())
        existing.updated_at = datetime.utcnow()

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return self.db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        更新用户偏好
        """
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                raise ValueError(f"用户 {user_id} 画像不存在")

            # 更新行业偏好
            if 'industry_preferences' in preferences:
                profile.industry_preferences = preferences['industry_preferences']

            # 更新投资偏好
            if 'investment_preferences' in preferences:
                profile.investment_preferences = preferences['investment_preferences']

            # 重新构建20维向量
            updated_vector = profile.build_vector_from_components()
            profile.set_profile_vector_20d(updated_vector)
            profile.updated_at = datetime.utcnow()

            self.db.commit()
            logger.info(f"用户 {user_id} 偏好更新成功")

            return True

        except Exception as e:
            logger.error(f"更新用户 {user_id} 偏好失败: {str(e)}")
            self.db.rollback()
            return False

    def create_custom_user_profile(self, user_id: str,
                                   industry_preferences: List[float],
                                   investment_preferences: List[float]) -> bool:
        """
        创建自定义用户画像
        """
        try:
            if len(industry_preferences) != 11:
                raise ValueError("行业偏好必须是11维")
            if len(investment_preferences) != 9:
                raise ValueError("投资偏好必须是9维")

            profile_data = {
                'industry_preferences': industry_preferences,
                'investment_preferences': investment_preferences
            }

            profile = self.init_user_profile(user_id, reset=True, profile_data=profile_data)
            return profile is not None

        except Exception as e:
            logger.error(f"创建自定义用户画像失败: {str(e)}")
            return False

    def get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """获取用户向量"""
        profile = self.get_user_profile(user_id)
        if profile:
            return profile.get_profile_vector_20d()
        return None

    def get_profile_details(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像详情"""
        profile = self.get_user_profile(user_id)
        if not profile:
            return None

        return {
            "user_id": profile.user_id,
            "industry_preferences": profile.industry_preferences,
            "investment_preferences": profile.investment_preferences,
            "vector_20d": profile.get_profile_vector_20d().tolist(),
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat()
        }

    def update_user_behavior(self, user_id: str, behavior_data: Dict[str, Any]):
        """
        更新用户行为数据（简化版，基于股票交互）
        """
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                raise ValueError(f"用户 {user_id} 画像不存在")

            behavior_type = behavior_data.get('type', 'click')
            stock_symbol = behavior_data.get('stock_symbol')
            stock_sector = behavior_data.get('stock_sector')

            # 基于用户行为微调偏好
            if stock_sector and stock_sector in self.sector_list:
                self._adjust_sector_preference(profile, stock_sector, behavior_type)

            # 重新构建向量
            updated_vector = profile.build_vector_from_components()
            profile.set_profile_vector_20d(updated_vector)
            profile.updated_at = datetime.utcnow()

            # 显式标记JSON字段已被修改，以便SQLAlchemy能检测到变化，他只能自动更新除json外的字段
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(profile, 'industry_preferences')
            flag_modified(profile, 'investment_preferences')

            self.db.commit()
            logger.info(f"用户 {user_id} 行为更新成功: {behavior_type}")

            return True

        except Exception as e:
            logger.error(f"更新用户 {user_id} 行为失败: {str(e)}")
            self.db.rollback()
            return False

    def _adjust_sector_preference(self, profile: UserProfile, sector: str, behavior_type: str):
        """调整行业偏好"""
        try:
            sector_index = self.sector_list.index(sector)
            current_prefs = profile.industry_preferences or [0.5] * 11

            adjustment = 0.1 if behavior_type in ['click', 'favorite'] else -0.05
            new_pref = max(0, min(1, current_prefs[sector_index] + adjustment))

            current_prefs[sector_index] = new_pref
            profile.industry_preferences = current_prefs

        except ValueError:
            pass  # 行业不在预定义列表中

    async def update_profile_and_embed(self, user: UserInDB, profile: dict) -> None:
        text = "\n".join([f"{k}: {v}" for k, v in profile.items() if v])
        vec = (await self.embedder.embed([text or user.username], dim=self.dim))[0]
        await self.repo.update_profile_and_embedding(user.id, profile, vec)