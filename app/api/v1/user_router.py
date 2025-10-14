from __future__ import annotations
from fastapi import APIRouter, Depends, Query, HTTPException
from app.model.models import UserProfile, BehaviorEvent
from app.services.news_service import NewsService
from pydantic import BaseModel, Field
from requests.sessions import Session
from fastapi.security import OAuth2PasswordBearer
from app.model.models import UserPublic
from app.services.auth_service import AuthService
from app.deps import get_auth_service, get_user_service, get_stock_service
from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.database import Database

from app.adapters.db.database_client import get_postgres_session, get_mongo_db
from app.services.user_service import UserService
from app.deps import get_auth_service, get_user_service
from datetime import datetime, timezone
from app.services.stock_service import StockService
router = APIRouter(prefix="/users", tags=["users"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_service() -> NewsService:
    from app.main import svc
    return svc

router = APIRouter(prefix="/user", tags=["user"])

@router.post("/profile/init")
def init_profile(user_id: str = "demo",
                 reset: bool = Query(False, description="是否重置已有画像为零向量"),
                 service: NewsService = Depends(get_service)) -> UserProfile:
    prof = service.prof_repo.get_or_create(user_id)
    if reset:
        dim = getattr(service.embedder, "dim", len(prof.vector) or 32)
        prof.vector = [0.0]*dim
        service.prof_repo.save(prof)
    return prof

@router.post("/event")
def add_event(ev: BehaviorEvent, service: NewsService = Depends(get_service)):
    prof = service.record_event_and_update_profile(ev)
    return {"ok": True, "user_id": ev.user_id, "profile_normed_head": prof.vector[:5]}

class ClickEvent(BaseModel):
    user_id: str = "demo"
    news_id: str
    dwell_ms: int = 0
    liked: bool = False
    bookmarked: bool = False

class LikeEvent(BaseModel):
    user_id: str = "demo"
    news_id: str

class BookmarkEvent(BaseModel):
    user_id: str = "demo"
    news_id: str

@router.post("/event/click")
def user_click(ev: ClickEvent, service: NewsService = Depends(get_service)):
    # 取新闻向量（语义 + 画像）
    doc = service.news_repo.get(ev.news_id)
    if not doc:
        raise HTTPException(404, "news not found")

    news_sem = list(getattr(doc, "vector", []) or [])
    news_prof = list(getattr(doc, "vector_prof_20d", []) or [])

    # 简单权重：点击=1，>10s=1.5，like/bookmark 各+0.5
    w = 1.0
    if ev.dwell_ms >= 10000: w += 0.5
    if ev.liked: w += 0.5
    if ev.bookmarked: w += 0.5

class ProfileUpdateIn(BaseModel):
    full_name: str | None = None
    bio: str | None = None
    interests: list[str] = []
    sectors: list[str] = []
    tickers: list[str] = []

@router.get("/me", response_model=UserPublic)
async def me(
        token: str = Depends(oauth2_scheme),
        auth: AuthService = Depends(get_auth_service)):
    u = await auth.get_current_user(token)
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

#旧方法直接初始化用户画像，新方法在authrouter中，把pg中的用户向量和mongodb中的用户信息一起注册
# @router.post("/profile/init")
# async def init_user_profile(
#         user_id: str = Query(..., description="用户ID"),
#         reset: bool = Query(False, description="是否重置现有用户画像"),
#         user_service: UserService = Depends(get_user_service),
# ):
#     """
#     初始化20维用户画像
#     """
#     try:
#         profile = user_service.init_user_profile(user_id, reset)
#
#         if not profile:
#             raise HTTPException(status_code=500, detail="用户画像初始化失败")
#
#         vector_20d = profile.get_profile_vector_20d()
#
#         return {
#             "ok": True,
#             "user_id": user_id,
#             "message": "用户画像重置成功" if reset else "用户画像初始化成功",
#             "vector_dim": len(vector_20d),
#             "industry_preferences": profile.industry_preferences,
#             "investment_preferences": profile.investment_preferences,
#             "created_at": profile.created_at.isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"用户画像初始化失败: {str(e)}")


@router.post("/profile/custom")
async def create_custom_user_profile(
        user_id: str = Query(..., description="用户ID"),
        industry_preferences: str = Query(..., description="11维行业偏好，逗号分隔"),
        investment_preferences: str = Query(..., description="9维投资偏好，逗号分隔"),
        user_service: UserService = Depends(get_user_service),
):
    """
    创建自定义用户画像
    """
    try:
        # 解析偏好数据
        industry_prefs = [float(x.strip()) for x in industry_preferences.split(",")]
        investment_prefs = [float(x.strip()) for x in investment_preferences.split(",")]

        if len(industry_prefs) != 11:
            raise HTTPException(status_code=400, detail="行业偏好必须是11维")
        if len(investment_prefs) != 9:
            raise HTTPException(status_code=400, detail="投资偏好必须是9维")

        success = user_service.create_custom_user_profile(
            user_id, industry_prefs, investment_prefs
        )

        if not success:
            raise HTTPException(status_code=500, detail="自定义用户画像创建失败")

        return {
            "ok": True,
            "user_id": user_id,
            "message": "自定义用户画像创建成功",
            "industry_preferences": industry_prefs,
            "investment_preferences": investment_prefs
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"自定义用户画像创建失败: {str(e)}")


@router.get("/profile/detail")
async def get_user_profile_detail(
        user_id: str = Query(..., description="用户ID"),
        user_service: UserService = Depends(get_user_service),
):
    """
    获取用户画像详细信息
    """
    try:
        profile_details = user_service.get_profile_details(user_id)

        if not profile_details:
            raise HTTPException(status_code=404, detail="用户画像未找到")

        return {
            "ok": True,
            "profile": profile_details
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户画像详情失败: {str(e)}")


@router.get("/vector/{user_id}")
async def get_user_vector(
        user_id: str,
        user_service: UserService = Depends(get_user_service),
):
    """
    获取用户20维向量
    """
    try:
        user_vector = user_service.get_user_vector(user_id)

        if user_vector is None:
            raise HTTPException(status_code=404, detail="用户向量未找到")

        return {
            "ok": True,
            "user_id": user_id,
            "vector_20d": user_vector.tolist(),
            "vector_dim": len(user_vector)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户向量失败: {str(e)}")


@router.post("/behavior/update")
async def update_user_behavior(
        user_id: str = Query(..., description="用户ID"),
        behavior_type: str = Query(..., description="行为类型: click, favorite, dislike, unfavorite, undislike"),
        stock_symbol: str = Query(..., description="股票代码"),
        stock_sector: str = Query(None, description="股票行业，如果不提供尝试从MongoDB获取"),
        invest_update: bool = Query(True, description="是否更新投资偏好"),
        duration: float = Query(0, description="停留时间(秒)"),
        mongo_db: Database = Depends(get_mongo_db),
        user_service: UserService = Depends(get_user_service),
        stock_service: StockService = Depends(get_stock_service),
):
    """
    更新用户行为数据，用于动态调整用户偏好
    """
    try:

        behavior_data = {
            "type": behavior_type,
            "stock_symbol": stock_symbol,
            "stock_sector": stock_sector,
            "duration": duration,
            "invest_update": invest_update
        }

        #使用异步版本
        success = await user_service.update_user_behavior(user_id, behavior_data, mongo_db, stock_service)

        if not success:
            raise HTTPException(status_code=500, detail="用户行为更新失败")

        # 获取更新后的用户偏好
        profile_details = user_service.get_profile_details(user_id)

        return {
            "ok": True,
            "user_id": user_id,
            "behavior_type": behavior_type,
            "stock_symbol": stock_symbol,
            # "stock_sector": stock_sector, 需要让他更新
            "message": "用户行为数据更新成功",
            "updated_preferences": {
                "industry_preferences": profile_details["industry_preferences"],
                "investment_preferences": profile_details["investment_preferences"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"用户行为更新失败: {str(e)}")


@router.post("/preferences/update")
async def update_user_preferences(
        user_id: str = Query(..., description="用户ID"),
        industry_preferences: str = Query(None, description="11维行业偏好，逗号分隔"),
        investment_preferences: str = Query(None, description="9维投资偏好，逗号分隔"),
        # postgres_db: Session = Depends(get_postgres_session),
        user_service: UserService = Depends(get_user_service),
):
    """
    直接更新用户偏好
    """
    try:
        # user_service = UserService(postgres_db)

        preferences = {}

        # 解析行业偏好
        if industry_preferences:
            industry_prefs = [float(x.strip()) for x in industry_preferences.split(",")]
            if len(industry_prefs) != 11:
                raise HTTPException(status_code=400, detail="行业偏好必须是11维")
            preferences['industry_preferences'] = industry_prefs

        # 解析投资偏好
        if investment_preferences:
            investment_prefs = [float(x.strip()) for x in investment_preferences.split(",")]
            if len(investment_prefs) != 9:
                raise HTTPException(status_code=400, detail="投资偏好必须是9维")
            preferences['investment_preferences'] = investment_prefs

        if not preferences:
            raise HTTPException(status_code=400, detail="请提供要更新的偏好数据")

        success = user_service.update_user_preferences(user_id, preferences)

        if not success:
            raise HTTPException(status_code=500, detail="用户偏好更新失败")

        # 获取更新后的用户信息
        profile_details = user_service.get_profile_details(user_id)

        return {
            "ok": True,
            "user_id": user_id,
            "message": "用户偏好更新成功",
            "updated_profile": profile_details
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"用户偏好更新失败: {str(e)}")


@router.get("/preferences/explain")
async def explain_user_preferences(
        user_id: str = Query(..., description="用户ID"),
        # postgres_db: Session = Depends(get_postgres_session),
        user_service: UserService = Depends(get_user_service),
):
    """
    解释用户偏好含义
    """
    try:
        # user_service = UserService(postgres_db)
        profile_details = user_service.get_profile_details(user_id)

        if not profile_details:
            raise HTTPException(status_code=404, detail="用户画像未找到")

        industry_prefs = profile_details["industry_preferences"]
        investment_prefs = profile_details["investment_preferences"]

        # 行业偏好解释
        sector_explanations = []
        for i, pref in enumerate(industry_prefs):
            sector_name = user_service.sector_list[i]
            if pref > 0.7:
                level = "强烈偏好"
            elif pref > 0.6:
                level = "偏好"
            elif pref > 0.4:
                level = "中性"
            elif pref > 0.3:
                level = "不太偏好"
            else:
                level = "不偏好"
            sector_explanations.append({
                "sector": sector_name,
                "preference_level": level,
                "score": pref
            })

        # 投资偏好解释
        investment_descriptions = [
            "市值偏好 (0=小盘股, 1=大盘股)",
            "成长价值偏好 (0=价值型, 1=成长型)",
            "股息偏好 (0=不看重, 1=很看重)",
            "风险承受 (0=保守, 1=激进)",
            "流动性需求 (0=低流动性, 1=高流动性)",
            "质量偏好 (0=不看重, 1=很看重)",
            "估值安全偏好 (0=可接受高估值, 1=要求安全边际)",
            "动量偏好 (0=不追涨, 1=相信动量)",
            "效率偏好 (0=不关注, 1=很看重)"
        ]

        investment_explanations = []
        for i, (pref, desc) in enumerate(zip(investment_prefs, investment_descriptions)):
            investment_explanations.append({
                "dimension": desc,
                "score": pref,
                "interpretation": f"得分 {pref:.2f}"
            })

        return {
            "ok": True,
            "user_id": user_id,
            "sector_preferences": sector_explanations,
            "investment_preferences": investment_explanations
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解释用户偏好失败: {str(e)}")
    # 更新用户画像
    service.prof_repo.update_user_vectors_from_event(
        user_id=ev.user_id,
        news_sem=news_sem,
        news_prof=news_prof,
        weight=w,
    )

    service.ev_repo.add(BehaviorEvent(
        user_id=ev.user_id,
        news_id=ev.news_id,
        type="click",
        ts=datetime.now(timezone.utc),
        dwell_ms=ev.dwell_ms,
        liked=ev.liked,
        bookmarked=ev.bookmarked,
    ))
    return {"ok": True, "weight": w}

@router.post("/event/like")
def user_like(ev: LikeEvent, service: NewsService = Depends(get_service)):
    doc = service.news_repo.get(ev.news_id)
    if not doc:
        raise HTTPException(404, "news not found")
    news_sem = list(getattr(doc, "vector", []) or [])
    news_prof = list(getattr(doc, "vector_prof_20d", []) or [])

    # like 权重偏低，聚合“认同”信号
    w = 0.7
    service.prof_repo.update_user_vectors_from_event(
        user_id=ev.user_id, news_sem=news_sem, news_prof=news_prof, weight=w,
    )
    return {"ok": True, "weight": w}

@router.post("/event/bookmark")
def user_bookmark(ev: BookmarkEvent, service: NewsService = Depends(get_service)):
    doc = service.news_repo.get(ev.news_id)
    if not doc:
        raise HTTPException(404, "news not found")
    news_sem = list(getattr(doc, "vector", []) or [])
    news_prof = list(getattr(doc, "vector_prof_20d", []) or [])

    # bookmark 权重略高，表示“强偏好/留存”
    w = 1.2
    service.prof_repo.update_user_vectors_from_event(
        user_id=ev.user_id, news_sem=news_sem, news_prof=news_prof, weight=w,
    )
    return {"ok": True, "weight": w}