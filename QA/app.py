import logging
from logging.handlers import RotatingFileHandler
import warnings
import os
import time
import socket
import sys
from collections import OrderedDict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import src.config as cfg
from src.rag_engine import (
    build_vectorstore,
    retrieve,
    rewrite_query,
    multi_query_retrieve,
    need_search,
    NO_RESULT,
)

_handler = RotatingFileHandler("./app.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler, logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- 提示词常量 ---
CHAT_SYSTEM_PROMPT = "你是一个友好的聊天伙伴。回答简洁自然，善用 Markdown 格式提升可读性。"
QA_SYSTEM_PROMPT = """你是一个PDF问答助手。根据以下检索到的资料回答问题。

【严格规则 - 必须遵守】
1. 所有数字、百分比、金额、日期必须逐字引用原文，**不得修改、推算、编造**
2. 只引用资料中明确写明的信息，不要做推论（如"500人"不能推成"487人"）
3. 页码标注必须与实际来源页一致
4. 如果检索到的资料不包含答案，你可以根据对话历史中已经提到的信息回答，不要编造

=== 检索到的资料 ===
{context}"""

# --- 启动校验 ---
if not cfg.API_KEY:
    logger.error("API_KEY 未配置，请在 .env 中设置")
    sys.exit(1)


class LRUCache:
    """LRU + TTL 缓存，替代无限增长的 dict"""
    def __init__(self, maxsize: int, ttl: int):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, str] = OrderedDict()
        self._timestamps: dict[str, float] = {}

    def get(self, key: str) -> str | None:
        if key not in self._data:
            return None
        if time.time() - self._timestamps[key] > self.ttl:
            self._evict(key)
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: str):
        self._data[key] = value
        self._timestamps[key] = time.time()
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def _evict(self, key: str):
        self._data.pop(key, None)
        self._timestamps.pop(key, None)


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

    def _get_cached_answer(self, question: str):
        exact = self._answer_cache.get(question)
        if exact is not None:
            logger.info("缓存命中 (精确): %s", question[:40])
            return exact
        # 模糊匹配只在缓存条目数较少时进行，避免 SequenceMatcher 扫描全部
        if len(self._answer_cache._data) > 50:
            return None
        for q in self._answer_cache._data:
            if SequenceMatcher(None, question, q).ratio() > self._cache_sim_threshold:
                answer = self._answer_cache.get(q)  # 顺带刷新 LRU
                if answer:
                    logger.info("缓存命中 (模糊): %s ~ %s", question[:40], q[:40])
                    return answer
        return None

    def _build_history(self, history):
        """将 Gradio history 转为 LangChain 消息列表"""
        messages = []
        for msg in history:
            if isinstance(msg, dict):
                messages.append(
                    HumanMessage(content=msg["content"])
                    if msg.get("role") == "user"
                    else AIMessage(content=msg["content"])
                )
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                messages.append(
                    HumanMessage(content=str(msg[0]))
                    if str(msg[0]) != ""
                    else AIMessage(content=str(msg[1]))
                )
        return messages

    def load_pdfs(self, files):
        if not files:
            logger.warning("load_pdfs 收到空文件列表")
            return "请选择 PDF 文件"

        pdf_entries = []
        for f in files:
            path = f.name if hasattr(f, "name") else f
            name = os.path.basename(path)
            pdf_entries.append((path, name))

        logger.info("开始加载 %d 个 PDF: %s", len(pdf_entries), [n for _, n in pdf_entries])

        self._last_context = None

        try:
            self.vectorstore, summaries = build_vectorstore(self.llm, pdf_entries, self.vectorstore)
            loaded_str = "、".join(n for _, n in pdf_entries)
            # 展示摘要信息
            summary_str = ""
            if summaries:
                summary_str = " | " + " | ".join(summaries)
            # 清理 Gradio 临时文件
            for path, _ in pdf_entries:
                if path.startswith(os.environ.get("GRADIO_TEMP_DIR", "")):
                    try:
                        os.remove(path)
                        logger.info("已清理临时文件: %s", path)
                    except OSError:
                        pass
            return f"已加载: {loaded_str}{summary_str}"
        except Exception as e:
            logger.error("加载 PDF 失败: %s", e, exc_info=True)
            return f"加载失败: {e}"

    def respond_stream(self, message, history):
        logger.info("respond_stream 被调用, vectorstore=%s", self.vectorstore is not None)

        if not history:
            cached = self._get_cached_answer(message)
            if cached:
                yield cached
                return

        # 无文档 → 纯聊天
        if self.vectorstore is None:
            yield from self._stream_answer(CHAT_SYSTEM_PROMPT, message, history, None)
            return

        # 有文档 → 先判断是否与文档相关，再决定是否检索
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
                yield from self._stream_answer(QA_SYSTEM_PROMPT.format(context=context), message, history, message)
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

    def _stream_answer(self, system_prompt, message, history, cache_key):
        messages = [SystemMessage(content=system_prompt)] + self._build_history(history)
        messages.append(HumanMessage(content=message))
        full = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full += chunk.content
                yield chunk.content
        if cache_key and not history:
            self._answer_cache.put(cache_key, full)


engine = RAGEngine()


# ============ Gradio 界面 ============

_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "style.css")
try:
    with open(_css_path, encoding="utf-8") as _f:
        CSS = _f.read()
except FileNotFoundError:
    CSS = ""

import gradio as gr


def initialize():
    engine.vectorstore = None
    logger.info("初始化完成，等待上传 PDF")
    return "请上传 PDF 文件开始问答"


with gr.Blocks(title="PDF 智能问答系统", css=CSS, theme=gr.themes.Soft(
    font=gr.themes.GoogleFont("Inter"),
    primary_hue="indigo",
    neutral_hue="slate",
)) as demo:
    gr.HTML("""
    <div class="top-bar">
      <div class="top-bar-left">
        <h1>PDF 智能问答</h1>
        <span class="badge">v2</span>
      </div>
      <p>上传文档 · 智能解析 · 即刻问答</p>
    </div>
    """)

    with gr.Row(equal_height=False, elem_id="main-layout"):
        # 左侧：文档管理
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<div class="section-label">📄 文档管理</div>')
            status_display = gr.Textbox(
                label="", interactive=False, show_label=False,
                container=True, elem_id="status-display",
            )
            file_input = gr.File(
                label="", file_types=[".pdf"],
                show_label=False, container=False, file_count="multiple",
            )
            gr.HTML('<div class="upload-hint">📄 点击选择或拖拽 PDF 文件到上方区域</div>')
            upload_btn = gr.Button("加载文档", variant="primary", size="sm")
            gr.HTML("""
            <div style="margin-top:16px;">
              <div class="section-label">💡 使用提示</div>
              <div style="display:flex;flex-direction:column;gap:6px;font-size:12.5px;color:#71717a;line-height:1.6;">
                <span>📎 可上传多个 PDF，自动建库统一检索</span>
                <span>💬 普通聊天也可，系统自动判断是否查文档</span>
                <span>🔍 回答标注来源文档和页码，方便核对原文</span>
              </div>
            </div>
            """)
            with gr.Row():
                status_display.value = initialize()

        # 右侧：对话
        with gr.Column(scale=2, elem_classes="chat-col"):
            gr.HTML('<div class="chat-header">'
                    '<span class="chat-header-label">💬 对话</span>'
                    '<span class="chat-header-meta">系统就绪</span></div>')
            thinking = gr.HTML(
                '<div class="thinking-dots">'
                '<div class="dots"><span></span><span></span><span></span></div>'
                '<span class="label">AI 思考中…</span></div>',
                visible=False, elem_id="thinking-indicator",
            )
            chatbot = gr.Chatbot(
                label="", height=None, show_label=False,
                render_markdown=True, scale=1, elem_classes="chatbot-area",
            )
            gr.HTML('<div class="chat-divider"></div>')
            with gr.Group(elem_id="chat-input-area"):
                with gr.Row():
                    msg = gr.Textbox(
                        label="", placeholder="向文档提问，或随便聊聊…",
                        container=True, scale=9, show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80, elem_id="send-btn")
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot], value="清空对话", size="sm")
                    clear.variant = "secondary"

    def show_thinking():
        return gr.update(visible=True)

    def hide_thinking():
        return gr.update(visible=False)

    def respond_wrapper(message, chat_history):
        try:
            if not message.strip():
                yield "", chat_history
                return
            logger.info("用户消息: %s", message)
            # 拷贝一次 history，后续原地修改避免每轮复制
            chat_history = list(chat_history)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": ""})
            entry = chat_history[-1]
            for chunk in engine.respond_stream(message, chat_history):
                entry["content"] += chunk
                yield "", chat_history
            logger.info("助手回复完成, 长度: %d", len(entry["content"]))
        except Exception as e:
            logger.error("respond_wrapper 异常: %s", e, exc_info=True)
            # 回退正常流程已追加的空条目，避免重复
            if len(chat_history) >= 2 and chat_history[-1].get("content", "") == "":
                chat_history = chat_history[:-2]
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": f"系统错误: {e}"})
            yield "", chat_history

    msg.submit(show_thinking, None, thinking).then(
        respond_wrapper, [msg, chatbot], [msg, chatbot]
    ).then(hide_thinking, None, thinking)

    send_btn.click(show_thinking, None, thinking).then(
        respond_wrapper, [msg, chatbot], [msg, chatbot]
    ).then(hide_thinking, None, thinking)

    def handle_upload(files):
        if not files:
            logger.warning("上传文件为空")
            return "请选择 PDF 文件"
        logger.info("用户上传 %d 个文件", len(files))
        return engine.load_pdfs(files)

    upload_btn.click(
        lambda: "正在解析文档并生成向量库，请稍候…", None, status_display
    ).then(handle_upload, inputs=file_input, outputs=status_display)


def _acquire_process_lock(port: int) -> None:
    """通过端口绑定检测是否已有实例在运行，防止进程堆积。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        sock.close()
    except OSError:
        logger.error("端口 %d 已被占用，进程可能已在运行。退出。", port)
        sys.exit(1)


if __name__ == "__main__":
    _acquire_process_lock(cfg.APP_PORT)

    print("\n  -> 访问地址: http://127.0.0.1:%d\n" % cfg.APP_PORT)
    demo.launch(inbrowser=False, quiet=True, show_error=True)
