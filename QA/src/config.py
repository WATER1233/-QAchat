import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-flash")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
CHROMA_DIR = "./chroma_db"

DOC_QUESTION_KEYWORDS = [
    "pdf", "文档", "文件", "资料", "内容", "讲了", "提到", "关于",
    "总结", "概括", "概述", "摘要", "主题", "话题", "这篇", "这个",
    "简历", "resume", "自我评价", "项目经历", "技能", "优势", "评价",
    "概况", "介绍",
]
