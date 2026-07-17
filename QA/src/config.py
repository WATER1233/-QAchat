import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-flash")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
CHROMA_DIR = "./chroma_db"

# Reranker 配置
RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

# 文档分块
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- 企业级资源管控 ---

# Embedding 分批：每批最多处理多少条文本，减少峰值内存
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Embedding 最大并发数（跨批次）
EMBEDDING_MAX_WORKERS = int(os.getenv("EMBEDDING_MAX_WORKERS", "4"))

# 回答缓存：最多缓存条目数，超限淘汰最早条目
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "100"))

# 回答缓存：条目存活秒数，过期自动失效
CACHE_TTL = int(os.getenv("CACHE_TTL", "1800"))

# 进程锁端口：用于检测是否已有实例在运行
APP_PORT = int(os.getenv("APP_PORT", "7860"))
