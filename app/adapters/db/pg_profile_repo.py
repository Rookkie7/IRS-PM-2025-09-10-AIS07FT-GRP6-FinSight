from __future__ import annotations
import ast
import psycopg
import json
import math
from psycopg.rows import dict_row
from app.domain.models import UserProfile
import pgvector.psycopg as pgvector_psycopg
from psycopg.types.json import Json

DEFAULT_DIM_SEM = 64   # 语义向量维度（与你的 embedder / SRP 一致）
DEFAULT_DIM_PROF = 20  # 行业/偏好 20 维

import logging
log = logging.getLogger("app.pg_profile_repo")

def _zero(n: int) -> list[float]:
    return [0.0] * n

def _clip01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def _split20(vec20: list[float]) -> tuple[list[float], list[float]]:
    """分成 (行业11, 投资9)。长度不足补0，超出截断。"""
    v = list(vec20 or [])
    if len(v) < 20: v = v + [0.0]*(20-len(v))
    if len(v) > 20: v = v[:20]
    return v[:11], v[11:]

def _join20(ind11: list[float], inv9: list[float]) -> list[float]:
    ind11 = (ind11 or [])[:11]
    inv9  = (inv9  or [])[:9]
    if len(ind11) < 11: ind11 += [0.0]*(11-len(ind11))
    if len(inv9)  < 9:  inv9  += [0.0]*(9-len(inv9))
    return [float(_clip01(x)) for x in (ind11 + inv9)]

def _argmax_idx(xs: list[float], thr: float = 1e-6) -> int | None:
    if not xs: return None
    k = max(range(len(xs)), key=lambda i: xs[i])
    return k if xs[k] > thr else None

class PgProfileRepo:
    def __init__(self, dsn: str, dim: int):
        self.dsn = dsn
        # dim 只在老的单列 `vector(dim)` 用得到；新方案用固定 64/20 三列
        self.dim = int(dim) if dim else DEFAULT_DIM_SEM
        self._ensure_schema()

    def _conn(self):
        conn = psycopg.connect(self.dsn, row_factory=dict_row)
        pgvector_psycopg.register_vector(conn)  # 让 pgvector 向量与 Python 列表互转
        return conn

    def ping_detail(self) -> tuple[bool, str | None]:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True, None
        except Exception as e:
            return False, f"{e.__class__.__name__}: {e}"

    def ping(self) -> bool:
        ok, _ = self.ping_detail()
        return ok

    @staticmethod
    def _to_list(vec):
        """把 PG 取回的向量（pgvector.Vector/list/tuple/字符串）统一成 List[float]"""
        if vec is None:
            return None
        if isinstance(vec, (list, tuple)):
            return list(vec)
        if hasattr(vec, "to_list"):
            return vec.to_list()
        if isinstance(vec, str):
            try:
                return list(ast.literal_eval(vec))
            except Exception:
                s = vec.strip("[](){}")
                return [float(x) for x in s.split(",") if x.strip()]
        return list(vec)

    def _ensure_schema(self):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 按你的要求：20d 作为 TEXT（JSON 字符串）
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                vector vector({self.dim}),
                user_semantic_64d_short vector({DEFAULT_DIM_SEM}),
                user_semantic_64d_long  vector({DEFAULT_DIM_SEM}),
                profile_vector_20d      TEXT,            -- ★ 这里是 TEXT
                industry_preferences    JSON,            -- 保留
                investment_preferences  JSON,            -- 保留
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            );
            """)

            # 兜底补列（重复执行安全）
            cur.execute(f"ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS vector vector({self.dim});")
            cur.execute(f"ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS user_semantic_64d_short vector({DEFAULT_DIM_SEM});")
            cur.execute(f"ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS user_semantic_64d_long  vector({DEFAULT_DIM_SEM});")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS profile_vector_20d TEXT;")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS industry_preferences JSON;")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS investment_preferences JSON;")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();")

            # 存储已施加的补丁（只为可撤销的偏好类行为：like/bookmark）
            cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profile_patches (
                user_id TEXT NOT NULL,
                news_id TEXT NOT NULL,
                action  TEXT NOT NULL,                -- 'like' | 'bookmark'
                delta_prof REAL[],                    -- 20d 补丁（本次加到 profile_vector_20d 的增量）
                created_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (user_id, news_id, action)
            );
            """)

            conn.commit()

    def get_or_create(self, user_id: str) -> UserProfile:
        z64 = _zero(DEFAULT_DIM_SEM)
        z20 = _zero(DEFAULT_DIM_PROF)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT user_id, vector, user_semantic_64d_short, user_semantic_64d_long, profile_vector_20d
                FROM user_profiles WHERE user_id=%s
            """, (user_id,))
            row = cur.fetchone()

            if row:
                base_vec = self._to_list(row["vector"]) or _zero(self.dim)
                need_fix = (
                    row["user_semantic_64d_short"] is None or
                    row["user_semantic_64d_long"]  is None or
                    row["profile_vector_20d"]      is None
                )
                if need_fix:
                    cur.execute("""
                        UPDATE user_profiles
                        SET user_semantic_64d_short = COALESCE(user_semantic_64d_short, %s),
                            user_semantic_64d_long  = COALESCE(user_semantic_64d_long,  %s),
                            profile_vector_20d      = COALESCE(profile_vector_20d,      %s),
                            updated_at = now()
                        WHERE user_id=%s
                    """, (z64, z64, self._dump_prof20(z20), user_id))
                    conn.commit()
                return UserProfile(user_id=row["user_id"], vector=base_vec)

            base_vec = _zero(self.dim)
            cur.execute("""
                INSERT INTO user_profiles(user_id, vector, user_semantic_64d_short, user_semantic_64d_long, profile_vector_20d)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id, vector
            """, (user_id, base_vec, z64, z64, self._dump_prof20(z20)))
            r = cur.fetchone()
            conn.commit()
            return UserProfile(user_id=r["user_id"], vector=self._to_list(r["vector"]) or _zero(self.dim))

    def save(self, prof: UserProfile):
        """老接口：仅更新单列 vector。"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE user_profiles SET vector=%s, updated_at=now() WHERE user_id=%s",
                (prof.vector, prof.user_id)
            )
            conn.commit()

    def get_user_vectors(self, user_id: str) -> dict:
        z64 = _zero(DEFAULT_DIM_SEM)
        z20 = _zero(DEFAULT_DIM_PROF)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT user_semantic_64d_short, user_semantic_64d_long, profile_vector_20d
                FROM user_profiles WHERE user_id=%s
            """, (user_id,))
            row = cur.fetchone()

            if not row:
                return {"short": z64, "long": z64, "prof20": z20}

            short = self._to_list(row["user_semantic_64d_short"]) if row["user_semantic_64d_short"] is not None else z64
            long  = self._to_list(row["user_semantic_64d_long"])  if row["user_semantic_64d_long"]  is not None else z64
            prof  = self._norm_prof20(row["profile_vector_20d"])                             # ★ TEXT→20d

            # 如果 TEXT 无法解析或维度不对，顺手修复回库（只修 TEXT 列）
            if row["profile_vector_20d"] is None or len(prof) != DEFAULT_DIM_PROF:
                with self._conn() as conn2, conn2.cursor() as cur2:
                    cur2.execute("""
                        UPDATE user_profiles
                        SET profile_vector_20d = %s, updated_at = now()
                        WHERE user_id=%s
                    """, (self._dump_prof20(prof), user_id))
                    conn2.commit()

            return {"short": short, "long": long, "prof20": prof}

    def update_user_vectors_from_event(
        self,
        user_id: str,
        news_sem: list[float],
        news_prof: list[float],
        weight: float = 1.0,
        alpha_short: float = 0.4,
        alpha_long: float = 0.1,
        alpha_ind: float = 0.25,   # 行业 (11d)
        alpha_inv: float = 0.18,   # 投资偏好 (9d)
    ):
        # 读取现有三路
        cur_vals = self.get_user_vectors(user_id)
        u_s = cur_vals["short"]  or [0.0]*DEFAULT_DIM_SEM
        u_l = cur_vals["long"]   or [0.0]*DEFAULT_DIM_SEM

        # 组件：若库里 JSON 为空，给到 11/9 维零向量
        ind = self._read_json_array(user_id, "industry_preferences", 11)  # 见下方工具
        inv = self._read_json_array(user_id, "investment_preferences", 9)

        # —— 语义通道：对新闻 64d 做 L2 归一后加权 —— #
        def _l2(x):
            import math
            n = math.sqrt(sum(t*t for t in x)) or 1.0
            return [t/n for t in x]

        xs = [t * weight for t in _l2(news_sem or [0.0]*DEFAULT_DIM_SEM)]

        # —— 画像通道：前 11=行业，后 9=偏好；不归一化，只做 EMA —— #
        ni = (news_prof or [0.0]*DEFAULT_DIM_PROF)[:11]
        nv = (news_prof or [0.0]*DEFAULT_DIM_PROF)[11:]

        # 允许新闻带 soft-one-hot（多行业会分摊）；同时乘以行为权重
        ni = [t * weight for t in ni]
        nv = [t * weight for t in nv]

        # EMA
        def _ema(u, x, a):
            return [(1.0 - a)*ui + a*xi for ui, xi in zip(u, x)]

        u_s2 = _ema(u_s, xs, alpha_short)
        u_l2 = _ema(u_l, xs, alpha_long)
        ind2 = _ema(ind, ni, alpha_ind)
        inv2 = _ema(inv, nv, alpha_inv)

        # 简单裁剪到 [0,1]，避免越界抖动；**不做整向量归一化**
        def _clip01(v): 
            return [0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x)) for x in v]
        ind2 = _clip01(ind2)
        inv2 = _clip01(inv2)

        # profile_vector_20d TEXT 镜像：直接拼接 20 个数的 JSON 字符串（**不做归一化**）
        prof20_txt = json.dumps(ind2 + inv2)

        up_sql = """
        INSERT INTO user_profiles (
            user_id,
            user_semantic_64d_short, user_semantic_64d_long,
            industry_preferences, investment_preferences,
            profile_vector_20d, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (user_id) DO UPDATE SET
            user_semantic_64d_short = EXCLUDED.user_semantic_64d_short,
            user_semantic_64d_long  = EXCLUDED.user_semantic_64d_long,
            industry_preferences    = EXCLUDED.industry_preferences,
            investment_preferences  = EXCLUDED.investment_preferences,
            profile_vector_20d      = EXCLUDED.profile_vector_20d,
            updated_at              = now();
        """
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                up_sql,
                (user_id, u_s2, u_l2, Json(ind2), Json(inv2), prof20_txt)
            )
            conn.commit()

        return {"ok": True}

    # —— 小工具：读 JSON 组件列；无则返回定长零向量 —— #
    def _read_json_array(self, user_id: str, col: str, dim: int) -> list[float]:
        sql = f"SELECT {col} FROM user_profiles WHERE user_id=%s"
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
        if not row or row[col] is None:
            return [0.0]*dim
        try:
            v = row[col]  # psycopg3 会把 json 列 decode 为 Python 对象
            if not isinstance(v, list):
                return [0.0]*dim
            out = [float(x) for x in v][:dim]
            if len(out) < dim:
                out += [0.0]*(dim - len(out))
            return out
        except Exception:
            return [0.0]*dim

    def _norm_prof20(self, val) -> list[float]:
        """
        接受：None / str(JSON) / list / tuple / pgvector.Vector
        返回：list[float] 长度=20（不足补0，过长截断）
        """
        if val is None:
            v = []
        elif isinstance(val, str):
            try:
                v = json.loads(val)
            except Exception:
                v = []
        elif hasattr(val, "to_list"):
            v = val.to_list()
        elif isinstance(val, (list, tuple)):
            v = list(val)
        else:
            try:
                v = list(val)
            except Exception:
                v = []

        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                out.append(0.0)
        if len(out) < DEFAULT_DIM_PROF:
            out += [0.0] * (DEFAULT_DIM_PROF - len(out))
        elif len(out) > DEFAULT_DIM_PROF:
            out = out[:DEFAULT_DIM_PROF]
        return out

    def _dump_prof20(self, v20: list[float]) -> str:
        """把 20 维 list 序列化为 JSON 字符串写入 TEXT 列"""
        # 保险再裁剪/补齐一次
        vv = (v20 or [])[:DEFAULT_DIM_PROF]
        if len(vv) < DEFAULT_DIM_PROF:
            vv += [0.0] * (DEFAULT_DIM_PROF - len(vv))
        return json.dumps([float(x) for x in vv])
    
    def _clip01(self, arr):
        # 护栏：把 20d 限制在 [0,1]，避免出现负数或>1
        return [0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x)) for x in arr]

    def _ema(self, u: list[float], x: list[float], a: float) -> list[float]:
        return [(1.0 - a) * ui + a * xi for ui, xi in zip(u, x)]

    def _read_prof20_safe(self, raw) -> list[float]:
        """把 PG 读出的 profile_vector_20d（可能是 list/tuple/pgvector/str/json）统一成 list[float] 长度20。"""
        v = self._to_list(raw)  # 你已有的万能转列表
        if v is None:
            v = []
        try:
            v = [float(x) for x in v]
        except Exception:
            try:
                # 可能是 JSON/text
                v = json.loads(raw) if isinstance(raw, str) else []
                v = [float(x) for x in v]
            except Exception:
                v = []
        if len(v) < 20: v = v + [0.0]*(20-len(v))
        if len(v) > 20: v = v[:20]
        return v

    # def _write_prof20(self, cur, user_id: str, vec20: list[float]):
    #     """兼容两种列类型写回：优先尝试直接写数组，失败再写 JSON/text。"""
    #     vec20 = [float(_clip01(x)) for x in (vec20 or [])]
    #     if len(vec20) < 20: vec20 += [0.0]*(20-len(vec20))
    #     if len(vec20) > 20: vec20 = vec20[:20]

    #     up_sql_arr = """
    #     UPDATE user_profiles
    #     SET profile_vector_20d = %s,
    #         updated_at = now()
    #     WHERE user_id = %s
    #     """
    #     try:
    #         cur.execute(up_sql_arr, (vec20, user_id))  # 如果列是 real[]，这一步就成功
    #         return
    #     except Exception as e:
    #         log.debug(f"[prof20] array-write failed, fallback to json text: {e.__class__.__name__}: {e}")

    #     up_sql_json = """
    #     UPDATE user_profiles
    #     SET profile_vector_20d = %s,
    #         updated_at = now()
    #     WHERE user_id = %s
    #     """
    #     cur.execute(up_sql_json, (json.dumps(vec20), user_id))  # 列是 text/json 时走这里

    def _write_prof20(self, cur, user_id: str, vec20: list[float]):
        """强制写入 JSON 字符串格式，确保 TEXT 列永远存储为 '[]' 而非 '{}'。"""
        vec20 = [float(_clip01(x)) for x in (vec20 or [])]
        if len(vec20) < 20:
            vec20 += [0.0] * (20 - len(vec20))
        if len(vec20) > 20:
            vec20 = vec20[:20]

        json_str = json.dumps(vec20)  # ✅ 永远序列化为 JSON 数组字符串 "[]"
        up_sql = """
        UPDATE user_profiles
        SET profile_vector_20d = %s,
            updated_at = now()
        WHERE user_id = %s
        """
        cur.execute(up_sql, (json_str, user_id))

    def update_prof20_add(
        self,
        user_id: str,
        news_prof20: list[float],
        event_type: str,  # 'like' or 'save'
    ):
        """
        add：沿用你原始逻辑——分段 EMA
        - 行业11维：只更新命中的行业位（one-hot 最大那一维），其它行业位不动
        - 投资9维：按新闻的9维分量做 EMA
        """
        # α 按你之前的强弱：save > like
        if event_type == "save":
            alpha_ind, alpha_inv = 0.15, 0.08
        else:  # like
            alpha_ind, alpha_inv = 0.10, 0.06

        cur_vals = self.get_user_vectors(user_id)  # 你已有：返回 {"short":[64], "long":[64], "prof20":[20]}
        u20 = self._read_prof20_safe(cur_vals.get("prof20"))

        ind_u, inv_u = _split20(u20)
        news_ind, news_inv = _split20(news_prof20 or [])

        # 行业：只更新命中的那个索引
        k = _argmax_idx(news_ind, thr=1e-6)
        if k is not None:
            ind_u[k] = (1.0 - alpha_ind) * ind_u[k] + alpha_ind * 1.0  # 朝1.0靠拢
            ind_u[k] = _clip01(ind_u[k])

        # 投资：对九维逐一 EMA（以新闻分量作为目标值）
        for i in range(9):
            tgt = float(news_inv[i]) if i < len(news_inv) else 0.0
            if tgt < 0.0: tgt = 0.0
            if tgt > 1.0: tgt = 1.0
            inv_u[i] = (1.0 - alpha_inv) * inv_u[i] + alpha_inv * tgt
            inv_u[i] = _clip01(inv_u[i])

        new20 = _join20(ind_u, inv_u)

        with self._conn() as conn, conn.cursor() as cur:
            # 确保存在一行
            cur.execute("INSERT INTO user_profiles(user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
            self._write_prof20(cur, user_id, new20)
            conn.commit()

        log.debug(f"[prof20.add] user={user_id} type={event_type} k_ind={k} α=({alpha_ind},{alpha_inv}) new_head={new20[:5]}")
        return {"ok": True, "k_ind": k, "alpha_ind": alpha_ind, "alpha_inv": alpha_inv}

    def update_prof20_remove(
        self,
        user_id: str,
        news_prof20: list[float],
        event_type: str,  # 'like' or 'save'
    ):
        """
        remove：只对“当初命中的维度”各自减固定 δ，逐维 clamp 到 [0,1]，不做任何全向量归一/重建。
        - 行业 δ：save > like
        - 投资 δ：save > like
        """
        if event_type == "save":
            delta_ind, delta_inv = 0.12, 0.06
        else:  # like
            delta_ind, delta_inv = 0.08, 0.04

        cur_vals = self.get_user_vectors(user_id)
        u20 = self._read_prof20_safe(cur_vals.get("prof20"))
        ind_u, inv_u = _split20(u20)
        news_ind, news_inv = _split20(news_prof20 or [])

        # 行业：只回退命中位
        k = _argmax_idx(news_ind, thr=1e-6)
        if k is not None:
            ind_u[k] = _clip01(ind_u[k] - delta_ind)

        # 投资：只回退“这次新闻真正命中的维度”，用一个小阈值判断
        for i in range(9):
            if i < len(news_inv) and news_inv[i] > 0.15:
                inv_u[i] = _clip01(inv_u[i] - delta_inv)

        new20 = _join20(ind_u, inv_u)

        with self._conn() as conn, conn.cursor() as cur:
            self._write_prof20(cur, user_id, new20)
            conn.commit()

        log.debug(f"[prof20.remove] user={user_id} type={event_type} k_ind={k} δ=({delta_ind},{delta_inv}) new_head={new20[:5]}")
        return {"ok": True, "k_ind": k, "delta_ind": delta_ind, "delta_inv": delta_inv}



