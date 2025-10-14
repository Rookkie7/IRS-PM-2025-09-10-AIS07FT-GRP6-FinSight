from __future__ import annotations
import ast
import psycopg
from psycopg.rows import dict_row
from app.domain.models import UserProfile
import pgvector.psycopg as pgvector_psycopg

DEFAULT_DIM_SEM = 64   # 语义向量维度（与你的 embedder / SRP 一致）
DEFAULT_DIM_PROF = 20  # 行业/偏好 20 维

def _zero(n: int) -> list[float]:
    return [0.0] * n

# class PgProfileRepo:
#     def __init__(self, dsn: str, dim: int):
#         self.dsn = dsn
#         self.dim = dim  # 与嵌入器维度一致
#         self._ensure_schema()          # ← 新增：启动时确保表存在
    
#     def ping_detail(self) -> tuple[bool, str | None]:
#         """返回 (ok, error)。ok=False 时 error 为异常字符串。"""
#         try:
#             with self._conn() as conn, conn.cursor() as cur:
#                 cur.execute("SELECT 1;")
#                 cur.fetchone()
#             return True, None
#         except Exception as e:
#             return False, f"{e.__class__.__name__}: {e}"

#     def ping(self) -> bool:
#         ok, _ = self.ping_detail()
#         return ok
        
#     def _ensure_schema(self):
#         with self._conn() as conn, conn.cursor() as cur:
#             # 扩展
#             cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
#             # 表（维度用当前 dim）
#             cur.execute(f"""
#                 CREATE TABLE IF NOT EXISTS user_profiles (
#                     user_id   TEXT PRIMARY KEY,
#                     vector    vector({self.dim}),
#                     updated_at TIMESTAMPTZ DEFAULT now()
#                 );
#             """)
#             conn.commit()

#     @staticmethod
#     def _to_list(vec):
#         """
#         将 PG 返回的向量规范为 List[float]：
#         - 如果是 pgvector.Vector 或 list/tuple：直接转 list
#         - 如果是字符串（例如 "[0.1, 0.2]"）：用 literal_eval 转
#         """
#         if isinstance(vec, (list, tuple)):
#             return list(vec)
#         # 某些环境可能返回 pgvector.Vector
#         if hasattr(vec, "to_list"):
#             return vec.to_list()
#         if isinstance(vec, str):
#             try:
#                 return list(ast.literal_eval(vec))
#             except Exception:
#                 # 最后兜底：去除括号后按逗号切
#                 s = vec.strip("[](){}")
#                 return [float(x) for x in s.split(",") if x.strip()]
#         # 默认直接包装
#         return list(vec)
    
#     def _conn(self):
#         conn = psycopg.connect(self.dsn, row_factory=dict_row)
#         pgvector_psycopg.register_vector(conn)  # 让 pgvector <-> Python 列表互转
#         return conn

#     def get_or_create(self, user_id: str) -> UserProfile:
#         with self._conn() as conn, conn.cursor() as cur:
#             cur.execute("SELECT user_id, vector FROM user_profiles WHERE user_id=%s", (user_id,))
#             row = cur.fetchone()
#             if row:
#                 vec = self._to_list(row["vector"])
#                 return UserProfile(user_id=row["user_id"], vector=vec)
#             zero = [0.0] * self.dim
#             cur.execute(
#                 "INSERT INTO user_profiles(user_id, vector) VALUES (%s, %s) RETURNING user_id, vector",
#                 (user_id, zero)
#             )
#             r = cur.fetchone()
#             conn.commit()
#             return UserProfile(user_id=r["user_id"], vector=self._to_list(r["vector"]))

#     def save(self, prof: UserProfile):
#         with self._conn() as conn, conn.cursor() as cur:
#             cur.execute(
#                 "UPDATE user_profiles SET vector=%s, updated_at=now() WHERE user_id=%s",
#                 (prof.vector, prof.user_id)
#             )
#             conn.commit()

#     def get_user_vectors(self, user_id: str):
#         sql = """
#         SELECT user_id,
#                user_semantic_64d_short,
#                user_semantic_64d_long,
#                user_profile_20d
#         FROM user_profiles
#         WHERE user_id = %s
#         """
#         self.cur.execute(sql, (user_id,))
#         row = self.cur.fetchone()
#         if not row:
#             return {
#                 "short": _zero(DEFAULT_DIM_SEM),
#                 "long":  _zero(DEFAULT_DIM_SEM),
#                 "prof20": _zero(DEFAULT_DIM_PROF),
#             }
#         return {
#             "short": row[1] or _zero(DEFAULT_DIM_SEM),
#             "long":  row[2] or _zero(DEFAULT_DIM_SEM),
#             "prof20": row[3] or _zero(DEFAULT_DIM_PROF),
#         }

#     def _ema(self, u, x, a):
#         # u, x: list[float], a in (0,1)
#         return [ (1-a)*ui + a*xi for ui,xi in zip(u, x) ]

#     def update_user_vectors_from_event(
#         self, user_id: str,
#         news_sem: list[float],
#         news_prof: list[float],
#         weight: float = 1.0,
#         alpha_short: float = 0.4,
#         alpha_long: float = 0.1,
#         alpha_prof: float = 0.15,
#     ):
#         # 读旧值
#         cur = self.get_user_vectors(user_id)
#         u_s = cur["short"]; u_l = cur["long"]; u_p = cur["prof20"]

#         # 归一 + 加权
#         def _l2(x):
#             import math
#             n = math.sqrt(sum(t*t for t in x)) or 1.0
#             return [t/n for t in x]
#         xs = [t*weight for t in _l2(news_sem)]
#         xp = news_prof  # 画像通道可以不归一，按权重EMA即可

#         # EMA
#         u_s2 = self._ema(u_s, xs, alpha_short)
#         u_l2 = self._ema(u_l, xs, alpha_long)
#         u_p2 = self._ema(u_p, xp, alpha_prof)

#         # 写回（upsert）
#         up_sql = """
#         INSERT INTO user_profiles(user_id, user_semantic_64d_short, user_semantic_64d_long, user_profile_20d)
#         VALUES (%s, %s, %s, %s)
#         ON CONFLICT (user_id) DO UPDATE
#            SET user_semantic_64d_short = EXCLUDED.user_semantic_64d_short,
#                user_semantic_64d_long  = EXCLUDED.user_semantic_64d_long,
#                user_profile_20d        = EXCLUDED.user_profile_20d,
#                updated_at = NOW()
#         """
#         self.cur.execute(up_sql, (user_id, u_s2, u_l2, u_p2))
#         self.conn.commit()
#         return {"ok": True}

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
        """兼容旧表：存在就补列；不存在就按新表建。"""
        with self._conn() as conn, conn.cursor() as cur:
            # pgvector 扩展
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 基表（新方案）
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                -- 旧字段：保留以兼容老接口（可选）
                vector vector({self.dim}),
                -- 新字段：两条 EMA 语义向量（64d）+ 行业/偏好 20d
                user_semantic_64d_short vector({DEFAULT_DIM_SEM}),
                user_semantic_64d_long  vector({DEFAULT_DIM_SEM}),
                user_profile_20d        real[],
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            );
            """)

            # 兜底：如果旧表里没有新列，则补列（重复执行也安全）
            cur.execute(f"ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS vector vector({self.dim});")
            cur.execute(f"ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS user_semantic_64d_short vector({DEFAULT_DIM_SEM});")
            cur.execute(f"ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS user_semantic_64d_long  vector({DEFAULT_DIM_SEM});")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS user_profile_20d        real[];")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();")
            cur.execute( "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();")

            conn.commit()

    # ------- 老接口（保留兼容）：vector 单列 -------

    # def get_or_create(self, user_id: str) -> UserProfile:
    #     """老接口：返回/创建一条仅含单列 vector 的 UserProfile（兼容现有代码）。"""
    #     with self._conn() as conn, conn.cursor() as cur:
    #         cur.execute("SELECT user_id, vector FROM user_profiles WHERE user_id=%s", (user_id,))
    #         row = cur.fetchone()
    #         if row:
    #             vec = self._to_list(row["vector"]) or _zero(self.dim)
    #             return UserProfile(user_id=row["user_id"], vector=vec)

    #         zero_vec = _zero(self.dim)
    #         cur.execute(
    #             "INSERT INTO user_profiles(user_id, vector) VALUES (%s, %s) RETURNING user_id, vector",
    #             (user_id, zero_vec)
    #         )
    #         r = cur.fetchone()
    #         conn.commit()
    #         return UserProfile(user_id=r["user_id"], vector=self._to_list(r["vector"]) or _zero(self.dim))

    def get_or_create(self, user_id: str) -> UserProfile:
        """
        兼容老接口：返回/创建一条仅含单列 vector 的 UserProfile。
        但在“创建”或“发现旧行有 NULL”时，会顺手把三路向量补成零向量，避免后续全是 NULL。
        """
        z64 = _zero(DEFAULT_DIM_SEM)
        z20 = _zero(DEFAULT_DIM_PROF)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT user_id, vector,
                    user_semantic_64d_short, user_semantic_64d_long, user_profile_20d
                FROM user_profiles WHERE user_id=%s
            """, (user_id,))
            row = cur.fetchone()

            if row:
                # 单列老向量兜底
                base_vec = self._to_list(row["vector"]) or _zero(self.dim)

                # 如果三路里有 NULL，顺手补零并落库
                need_fix = (
                    row["user_semantic_64d_short"] is None or
                    row["user_semantic_64d_long"]  is None or
                    row["user_profile_20d"]        is None
                )
                if need_fix:
                    cur.execute("""
                        UPDATE user_profiles
                        SET user_semantic_64d_short = COALESCE(user_semantic_64d_short, %s),
                            user_semantic_64d_long  = COALESCE(user_semantic_64d_long,  %s),
                            user_profile_20d        = COALESCE(user_profile_20d,        %s),
                            updated_at = now()
                        WHERE user_id=%s
                    """, (z64, z64, z20, user_id))
                    conn.commit()

                return UserProfile(user_id=row["user_id"], vector=base_vec)

            # 不存在则插入：单列老向量 + 三路都写零
            base_vec = _zero(self.dim)
            cur.execute("""
                INSERT INTO user_profiles(user_id, vector, user_semantic_64d_short, user_semantic_64d_long, user_profile_20d)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id, vector
            """, (user_id, base_vec, z64, z64, z20))
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

    # ------- 新接口：三路向量（短/长语义 64d + 画像 20d） -------

    # def get_user_vectors(self, user_id: str) -> dict:
    #     sql = """
    #     SELECT user_semantic_64d_short, user_semantic_64d_long, user_profile_20d
    #     FROM user_profiles
    #     WHERE user_id = %s
    #     """
    #     with self._conn() as conn, conn.cursor() as cur:
    #         cur.execute(sql, (user_id,))
    #         row = cur.fetchone()

    #     if not row:
    #         return {
    #             "short": _zero(DEFAULT_DIM_SEM),
    #             "long":  _zero(DEFAULT_DIM_SEM),
    #             "prof20": _zero(DEFAULT_DIM_PROF),
    #         }

    #     return {
    #         "short": self._to_list(row["user_semantic_64d_short"]) or _zero(DEFAULT_DIM_SEM),
    #         "long":  self._to_list(row["user_semantic_64d_long"])  or _zero(DEFAULT_DIM_SEM),
    #         "prof20": self._to_list(row["user_profile_20d"])       or _zero(DEFAULT_DIM_PROF),
    #     }

    def get_user_vectors(self, user_id: str) -> dict:
        """
        统一返回 dict：{"short":[..64..], "long":[..64..], "prof20":[..20..]}。
        - 如果行不存在：直接返回三路全 0（不落库，由首次事件或 init 接口创建）
        - 如果行存在但某路为 NULL：顺手把库里该路补成 0 并返回 0（防止后续接口读到 NULL）
        """
        z64 = _zero(DEFAULT_DIM_SEM)
        z20 = _zero(DEFAULT_DIM_PROF)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT user_semantic_64d_short, user_semantic_64d_long, user_profile_20d
                FROM user_profiles WHERE user_id=%s
            """, (user_id,))
            row = cur.fetchone()

            if not row:
                return {"short": z64, "long": z64, "prof20": z20}

            short = self._to_list(row["user_semantic_64d_short"]) if row["user_semantic_64d_short"] is not None else None
            long  = self._to_list(row["user_semantic_64d_long"])  if row["user_semantic_64d_long"]  is not None else None
            prof  = self._to_list(row["user_profile_20d"])        if row["user_profile_20d"]        is not None else None

            # 有 NULL 就就地补零并写回
            if short is None or long is None or prof is None:
                short = short or z64
                long  = long  or z64
                prof  = prof  or z20
                cur.execute("""
                    UPDATE user_profiles
                    SET user_semantic_64d_short = %s,
                        user_semantic_64d_long  = %s,
                        user_profile_20d        = %s,
                        updated_at = now()
                    WHERE user_id=%s
                """, (short, long, prof, user_id))
                conn.commit()

            return {"short": short, "long": long, "prof20": prof}
    
    @staticmethod
    def _ema(u: list[float], x: list[float], a: float) -> list[float]:
        # u, x: 同维向量；a∈(0,1)
        return [(1.0 - a) * ui + a * xi for ui, xi in zip(u, x)]

    def update_user_vectors_from_event(
        self,
        user_id: str,
        news_sem: list[float],
        news_prof: list[float],
        weight: float = 1.0,
        alpha_short: float = 0.4,
        alpha_long: float = 0.1,
        alpha_prof: float = 0.15,
    ):
        cur_vals = self.get_user_vectors(user_id)
        u_s = cur_vals["short"]
        u_l = cur_vals["long"]
        u_p = cur_vals["prof20"]

        # 语义通道：L2 归一 + 加权
        def _l2(x: list[float]) -> list[float]:
            import math
            n = math.sqrt(sum(t * t for t in x)) or 1.0
            return [t / n for t in x]

        xs = [t * weight for t in _l2(news_sem)]
        # 画像通道：可不归一，直接按权重 EMA
        xp = news_prof if news_prof else _zero(DEFAULT_DIM_PROF)

        # EMA 更新
        u_s2 = self._ema(u_s, xs, alpha_short)
        u_l2 = self._ema(u_l, xs, alpha_long)
        u_p2 = self._ema(u_p, xp, alpha_prof)

        up_sql = """
        INSERT INTO user_profiles (user_id, user_semantic_64d_short, user_semantic_64d_long, user_profile_20d, updated_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (user_id) DO UPDATE SET
            user_semantic_64d_short = EXCLUDED.user_semantic_64d_short,
            user_semantic_64d_long  = EXCLUDED.user_semantic_64d_long,
            user_profile_20d        = EXCLUDED.user_profile_20d,
            updated_at              = now();
        """
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(up_sql, (user_id, u_s2, u_l2, u_p2))
            conn.commit()

        return {"ok": True}