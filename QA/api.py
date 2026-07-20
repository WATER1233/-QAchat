import logging
import os
import sys
import time
import json
import asyncio
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

import src.config as cfg
from src.rag_engine import (
    build_vectorstore,
    multi_query_retrieve,
    need_search,
    rewrite_query,
    retrieve,
    NO_RESULT,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

if not cfg.API_KEY:
    logger.error("API_KEY 未配置，请在 .env 中设置")
    sys.exit(1)


# ===== 会话管理 =====

class SessionManager:
    """内存短期记忆：按 session_id 存储对话历史"""
    def __init__(self, max_sessions: int = 100):
        self.max_sessions = max_sessions
        self._sessions: dict[str, list[dict]] = OrderedDict()

    def get_or_create(self, session_id: str) -> list[dict]:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self._trim()
        return self._sessions[session_id]

    def append(self, session_id: str, role: str, content: str):
        history = self.get_or_create(session_id)
        history.append({"role": role, "content": content})
        self._sessions.move_to_end(session_id)

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def _trim(self):
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)


sessions = SessionManager()


# ===== 数据模型 =====

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    history: list[dict] | None = None  # 兼容旧客户端，可选


class StatusResponse(BaseModel):
    status: str
    message: str


# ===== RAG Engine =====

class LRUCache:
    def __init__(self, maxsize: int, ttl: int):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, str] = OrderedDict()
        self._timestamps: dict[str, float] = {}

    def get(self, key: str) -> str | None:
        if key not in self._data:
            return None
        if time.time() - self._timestamps[key] > self.ttl:
            self._data.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: str):
        self._data[key] = value
        self._timestamps[key] = time.time()
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)


CHAT_SYSTEM_PROMPT = "你是一个友好的聊天伙伴。回答简洁自然，善用 Markdown 格式提升可读性。"
QA_SYSTEM_PROMPT = """你是一个PDF问答助手。根据以下检索到的资料回答问题。

【严格规则 - 必须遵守】
1. 所有数字、百分比、金额、日期必须逐字引用原文，**不得修改、推算、编造**
2. 只引用资料中明确写明的信息，不要做推论（如"500人"不能推成"487人"）
3. 页码标注必须与实际来源页一致
4. 如果检索到的资料不包含答案，你可以根据对话历史中已经提到的信息回答，不要编造

=== 检索到的资料 ===
{context}"""


class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.llm = ChatOpenAI(
            base_url=cfg.BASE_URL,
            api_key=cfg.API_KEY,
            model=cfg.MODEL_NAME,
            temperature=0.3,
            max_tokens=1024,
        )
        self._last_context = None
        self._answer_cache = LRUCache(maxsize=cfg.CACHE_MAXSIZE, ttl=cfg.CACHE_TTL)
        self._cache_sim_threshold = 0.88

    def _get_cached_answer(self, question: str) -> str | None:
        exact = self._answer_cache.get(question)
        if exact is not None:
            logger.info("缓存命中 (精确): %s", question[:40])
            return exact
        if len(self._answer_cache._data) > 50:
            return None
        for q in self._answer_cache._data:
            if SequenceMatcher(None, question, q).ratio() > self._cache_sim_threshold:
                answer = self._answer_cache.get(q)
                if answer:
                    logger.info("缓存命中 (模糊): %s ~ %s", question[:40], q[:40])
                    return answer
        return None

    def _build_history(self, history: list[dict]):
        messages = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _stream_answer(self, system_prompt: str, message: str, history: list[dict], cache_key: str | None):
        messages = [SystemMessage(content=system_prompt)] + self._build_history(history)
        messages.append(HumanMessage(content=message))
        full = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full += chunk.content
                yield chunk.content
        if cache_key and not history:
            self._answer_cache.put(cache_key, full)

    def respond_stream(self, message: str, history: list[dict]):
        logger.info("respond_stream, vectorstore=%s", self.vectorstore is not None)

        if not history:
            cached = self._get_cached_answer(message)
            if cached:
                yield cached
                return

        if self.vectorstore is None:
            yield from self._stream_answer(CHAT_SYSTEM_PROMPT, message, history, None)
            return

        if not need_search(self.llm, message):
            logger.info("无需检索文档，直接聊天")
            if self._last_context:
                yield from self._stream_answer(
                    QA_SYSTEM_PROMPT.format(context=self._last_context),
                    message, history, None,
                )
            else:
                yield from self._stream_answer(CHAT_SYSTEM_PROMPT, message, history, None)
            return

        try:
            context = multi_query_retrieve(self.llm, self.vectorstore, message)
            if not context or context == NO_RESULT:
                rewritten = rewrite_query(self.llm, message)
                if rewritten != message:
                    context = retrieve(self.llm, self.vectorstore, rewritten)

            if context and context != NO_RESULT:
                self._last_context = context[:4000] if context else None
                yield from self._stream_answer(
                    QA_SYSTEM_PROMPT.format(context=context), message, history, message
                )
            elif self._last_context:
                logger.info("检索无结果，使用上一次上下文")
                yield from self._stream_answer(
                    QA_SYSTEM_PROMPT.format(context=self._last_context),
                    message, history, None,
                )
            else:
                yield from self._stream_answer(CHAT_SYSTEM_PROMPT, message, history, None)
        except Exception as e:
            logger.error("处理问题失败: %s", e, exc_info=True)
            yield f"错误: {e}"


_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


# ===== FastAPI App =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动")
    yield
    logger.info("服务关闭")


app = FastAPI(title="PDF 智能问答系统", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== API 路由 =====

@app.get("/api/health")
async def health():
    eng = get_engine()
    return {"status": "ok", "vectorstore_loaded": eng.vectorstore is not None}


@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """SSE 流式聊天（支持 session 记忆）"""
    eng = get_engine()
    history = sessions.get_or_create(request.session_id)

    # 如果前端传了 history，覆盖 session 已有数据（向后兼容）
    if request.history is not None:
        history[:] = request.history

    # 追加用户消息
    history.append({"role": "user", "content": request.message})

    def event_stream():
        full = ""
        try:
            for token in eng.respond_stream(request.message, history):
                full += token
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            # 追加 AI 回复到历史
            if full:
                sessions.append(request.session_id, "assistant", full)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error("SSE 流异常: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/new")
async def new_session():
    """创建新的空会话，返回 session_id"""
    import uuid
    session_id = uuid.uuid4().hex[:12]
    sessions.get_or_create(session_id)
    return {"session_id": session_id}


@app.get("/api/chat/{session_id}")
async def get_session_history(session_id: str):
    """获取指定 session 的对话历史"""
    history = sessions.get_or_create(session_id)
    return {"messages": history}


@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """上传 PDF 文件并建立向量索引"""
    if not files:
        raise HTTPException(status_code=400, detail="请选择 PDF 文件")

    temp_dir = Path(cfg.CHROMA_DIR).parent / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    pdf_entries = []
    saved_paths = []
    try:
        for f in files:
            if not f.filename or not f.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{f.filename} 不是 PDF 文件")

            save_path = temp_dir / f.filename
            content = await f.read()

            if len(content) > 200 * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"{f.filename} 超过 200MB 限制")

            save_path.write_bytes(content)
            saved_paths.append(save_path)
            pdf_entries.append((str(save_path), f.filename))

        logger.info("上传并保存 %d 个 PDF: %s", len(pdf_entries), [n for _, n in pdf_entries])

        eng = get_engine()
        eng._last_context = None
        vectorstore, summaries = build_vectorstore(eng.llm, pdf_entries, eng.vectorstore)
        eng.vectorstore = vectorstore

        loaded_str = "、".join(n for _, n in pdf_entries)
        summary_str = ""
        if summaries:
            summary_str = " | " + " | ".join(summaries)

        return StatusResponse(status="ok", message=f"已加载: {loaded_str}{summary_str}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("上传失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"加载失败: {e}")
    finally:
        for p in saved_paths:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


@app.get("/api/documents")
async def get_documents():
    """获取当前已加载的文档状态"""
    eng = get_engine()
    return {
        "loaded": eng.vectorstore is not None,
    }


@app.post("/api/documents/clear")
async def clear_documents():
    """清空文档索引"""
    eng = get_engine()
    eng.vectorstore = None
    eng._last_context = None
    logger.info("已清空文档索引")
    return StatusResponse(status="ok", message="已清空文档索引")


@app.post("/api/reset")
async def reset():
    """重置对话状态"""
    eng = get_engine()
    eng.vectorstore = None
    eng._last_context = None
    logger.info("已重置")
    return StatusResponse(status="ok", message="已重置")


# 生产模式：挂载前端构建产物
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="frontend_assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        content = (_frontend_dist / "index.html").read_text(encoding="utf-8")
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content)

    print("  -> 前端已集成，无需单独启动")
else:
    print("  -> 前端构建产物不存在，仅 API 模式运行")
    print("  -> 如需前端，先运行: cd frontend && npm run build")

if __name__ == "__main__":
    print(f"  -> 访问地址: http://127.0.0.1:{cfg.APP_PORT}\n")
    uvicorn.run(app, host="127.0.0.1", port=cfg.APP_PORT)
