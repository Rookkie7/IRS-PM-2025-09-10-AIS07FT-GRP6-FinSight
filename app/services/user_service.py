import numpy as np
import logging
from typing import List, Dict, Any, Optional
import random
import json
from fastapi import Depends
from datetime import datetime

from sqlalchemy.orm.session import Session
from sqlalchemy import text
from app.adapters.db.database_client import get_postgres_session
from app.model.models import UserProfile

from app.services.stock_service import StockService

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
    def __init__(self, db: Session,repo: UserRepoPort, dim: int = 20):
        self.db = db
        self.sector_list = SECTOR_LIST
        self.repo = repo
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

            # === 仅新增 64d 向量相关 ===
            zeros64 = json.dumps([0.0] * 64)  # 传字符串，由 SQL 层 ::vector 解析
            now = datetime.utcnow()

            if existing and reset:
                # 重置：强制写 0
                self.db.execute(
                    text("""
                        UPDATE user_profiles
                        SET user_semantic_64d_short = CAST(:v AS vector(64)),
                            user_semantic_64d_long  = CAST(:v AS vector(64)),
                            updated_at = :now
                        WHERE user_id = :uid
                    """),
                    {"v": zeros64, "now": now, "uid": user_id}
                )
            else:
                # 普通初始化：空时写 0（不覆盖已有值）
                self.db.execute(
                    text("""
                        UPDATE user_profiles
                        SET user_semantic_64d_short = COALESCE(user_semantic_64d_short, CAST(:v AS vector(64))),
                            user_semantic_64d_long  = COALESCE(user_semantic_64d_long,  CAST(:v AS vector(64))),
                            updated_at = :now
                        WHERE user_id = :uid
                    """),
                    {"v": zeros64, "now": now, "uid": user_id}
                )

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

    async def update_user_behavior(self, user_id: str, behavior_data: Dict[str, Any],mongo_db = None , stock_service = None):
        """
        更新用户行为数据方法（调整用户行业偏好）(调整用户投资偏好（可选））
        """
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                raise ValueError(f"用户 {user_id} 画像不存在")

            behavior_type = behavior_data.get('type', 'click')
            stock_symbol = behavior_data.get('stock_symbol')
            stock_sector = behavior_data.get('stock_sector')
            invest_update = behavior_data.get('invest_update')

            # 如果没有提供行业，尝试从MongoDB获取（异步）
            if not stock_sector and mongo_db is not None and stock_symbol:
                collection = mongo_db["stocks"]
                stock_data = await collection.find_one({"symbol": stock_symbol.upper()})
                if stock_data and 'basic_info' in stock_data:
                    stock_sector = stock_data['basic_info'].get('sector')
                    behavior_data['stock_sector'] = stock_sector

            # 基于用户行为微调偏好
            if invest_update == True:
                await self._adjust_preferences_based_on_behavior(profile, behavior_data,stock_service)
            elif stock_sector in self.sector_list:
                self._adjust_sector_preference_advanced(profile, stock_sector, behavior_type)

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

    async def _adjust_preferences_based_on_behavior(self, profile: UserProfile, behavior_data: Dict[str, Any],stock_service :StockService):
        """基于行为数据智能调整用户偏好"""
        behavior_type = behavior_data.get('type')
        duration = behavior_data.get('duration', 0)
        stock_sector = behavior_data.get('stock_sector')

        # 1. 行业偏好调整（考虑行为强度和时间）
        if stock_sector in self.sector_list:
            self._adjust_sector_preference_advanced(profile, stock_sector, behavior_type)

        # 2. 投资偏好调整（基于股票特征）
        stock_symbol = behavior_data.get('stock_symbol')
        if stock_symbol:
            await self._adjust_investment_preference_based_on_stock(profile, stock_symbol, behavior_type,stock_service)

    def _adjust_sector_preference_advanced(self, profile: UserProfile, sector: str,
                                           behavior_type: str):
        """行业偏好调整"""
        sector_index = self.sector_list.index(sector)
        current_prefs = profile.industry_preferences or [0.5] * 11

        # 基于行为类型的学习率
        learning_rates = {
            'click': 0.05,
            'favorite': 0.2,
            'dislike': -0.2,
            'unfavorite':-0.2,
            'undislike':0.2
        }

        learning_rate = learning_rates.get(behavior_type, 0)
        # 使用渐进式调整方法，向目标值1调整（对于正向行为）
        if learning_rate > 0:
            # 正向行为，向1调整
            target_value = 1.0
            current_prefs[sector_index] = max(0, min(1,
                                                     current_prefs[sector_index] + learning_rate * (
                                                                 target_value - current_prefs[sector_index])))
        elif learning_rate < 0:
            # 负向行为，向0调整
            target_value = 0.0
            current_prefs[sector_index] = max(0, min(1,
                                                     current_prefs[sector_index] + abs(learning_rate) * (
                                                                 target_value - current_prefs[sector_index])))
        # 如果learning_rate为0，则不调整
        profile.industry_preferences = current_prefs

    async def _adjust_investment_preference_based_on_stock(self, profile: UserProfile,
                                                           stock_symbol: str, behavior_type: str,
                                                           stock_service : StockService):
        """基于股票特征调整投资偏好"""
        try:
            # 获取股票的未归一化投资特征
            _, investment_features = await stock_service.compute_stock_vector_components(stock_symbol)
            current_prefs = profile.investment_preferences or [0.5] * 9

            # 根据行为类型调整：正反馈向股票特征靠近，负反馈远离
            if behavior_type == 'dislike':
                learning_rate = -0.2
            elif behavior_type == 'click':
                learning_rate = 0.05
            elif behavior_type == 'favorite':
                learning_rate = 0.2
            elif behavior_type == 'unfavorite':
                learning_rate = -0.2
            elif behavior_type == 'undislike':
                learning_rate = 0.2
            else:
                learning_rate = 0

            for i in range(9):
                current_prefs[i] = max(0, min(1,
                                              current_prefs[i] + learning_rate * (
                                                          investment_features[i] - current_prefs[i])))

            profile.investment_preferences = current_prefs

        except Exception as e:
            logger.error(f"调整投资偏好失败: {e}")

#这里这个方法在userrepo里没实现啊，用不了
    async def update_profile_and_embed(self, user: UserInDB, profile: dict) -> None:
        await self.repo.update_profile(user.id, profile)