import logging
import warnings
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import src.config as cfg
from src.embeddings import AliyunEmbeddings
from src.rag_engine import (
    build_vectorstore,
    retrieve,
    rewrite_query,
    remove_chroma_db,
    make_search_tool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.llm = ChatOpenAI(
            base_url=cfg.BASE_URL,
            api_key=cfg.API_KEY,
            model=cfg.MODEL_NAME,
            temperature=0.7,
            max_tokens=1024,
        )
        self.embeddings = AliyunEmbeddings()
        self.llm_with_tools = self.llm
        self._last_context = None   # 缓存上次检索结果，支持追问

    def update_tools(self):
        if self.vectorstore is not None:
            search_tool = make_search_tool(self.llm, self.vectorstore)
            self.llm_with_tools = self.llm.bind_tools([search_tool])
        else:
            self.llm_with_tools = self.llm

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

        global_vectorstore = getattr(self, "vectorstore", None)
        if global_vectorstore is not None:
            remove_chroma_db(cfg.CHROMA_DIR)

        self._last_context = None

        try:
            self.vectorstore, summaries = build_vectorstore(self.llm, pdf_entries)
            self.update_tools()
            loaded_str = "、".join(n for _, n in pdf_entries)
            for s in summaries:
                logger.info("摘要: %s", s)
            logger.info("PDF 加载完成: %s", loaded_str)
            return f"已加载: {loaded_str}"
        except Exception as e:
            logger.error("加载 PDF 失败: %s", e, exc_info=True)
            return f"加载失败: {e}"

    def respond_stream(self, message, history):
        logger.info("respond_stream 被调用, vectorstore=%s", self.vectorstore is not None)

        def build_history(history):
            msgs = []
            for turn in history:
                if isinstance(turn, dict):
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role == "user":
                        msgs.append(HumanMessage(content=content))
                    elif role == "assistant":
                        msgs.append(AIMessage(content=content))
            return msgs

        CHAT_SYSTEM_PROMPT = "你是一个友好的聊天伙伴。回答简洁自然，善用 Markdown 格式提升可读性。"

        # 纯聊天模式
        if self.vectorstore is None:
            messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + build_history(history)
            messages.append(HumanMessage(content=message))
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
            return

        # PDF 问答模式
        PDF_SYSTEM_PROMPT = """你是一个智能PDF问答助手。用户已上传PDF文档，你可以随时通过 SearchPDF 工具查询其内容。

【工作方式】
- 当用户**可能**在问文档内容时（包括：概况、介绍、公司信息、数字、数据、表格、财务、总结、文档讲了什么等），**必须**调用 SearchPDF 工具检索后回答，不要凭自己知识回答
- 只有确认是纯闲聊（问候、天气、你是谁等）才直接回答
- 不确定时优先调工具查文档
- 回答简洁准确，引用来源时标注 [文件名·第X页]"""

        QA_SYSTEM_PROMPT = """你是一个PDF问答助手。根据以下检索到的资料回答问题。

【严格规则 - 必须遵守】
1. 所有数字、百分比、金额、日期必须逐字引用原文，**不得修改、推算、编造**
2. 如果资料中不包含用户问的数据，直接回答"文档中没有提及"，不要猜测
3. 只引用资料中明确写明的信息，不要做推论（如"500人"不能推成"487人"）
4. 页码标注必须与实际来源页一致
5. 宁可少说，不要多说

=== 检索到的资料 ===
{context}"""

        try:
            # 策略1: 让 LLM 判断是否调 SearchPDF 工具
            pdf_messages = [SystemMessage(content=PDF_SYSTEM_PROMPT)] + build_history(history)
            pdf_messages.append(HumanMessage(content=message))
            response = self.llm_with_tools.invoke(pdf_messages)

            context = None
            tool_was_called = bool(response.tool_calls)

            if tool_was_called:
                logger.info("模型主动调用了 SearchPDF 工具")
                tool_call = response.tool_calls[0]
                query = tool_call["args"].get("query", message)
                logger.info("检索词: %s", query)
                search_tool = make_search_tool(self.llm, self.vectorstore)
                context = search_tool.invoke(query)

                if not context or context == "未检索到相关内容":
                    logger.info("工具检索无结果，尝试查询改写")
                    rewritten = rewrite_query(self.llm, message)
                    if rewritten != message:
                        context = retrieve(self.llm, self.vectorstore, rewritten)

            if context and context != "未检索到相关内容":
                self._last_context = context
                final_messages = [
                    SystemMessage(content=QA_SYSTEM_PROMPT.format(context=context)),
                ] + build_history(history) + [HumanMessage(content=message)]
                for chunk in self.llm.stream(final_messages):
                    if chunk.content:
                        yield chunk.content
            else:
                logger.info("工具未调用或检索无结果，走普通聊天模式")
                chat_messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + build_history(history)
                # 追问：注入上次检索到的上下文，让 LLM 能继续深入对话
                if self._last_context:
                    chat_messages.insert(1, SystemMessage(
                        content=f"以下是与该对话相关的文档资料，如需引用请标注来源：\n{self._last_context}"
                    ))
                chat_messages.append(HumanMessage(content=message))
                for chunk in self.llm.stream(chat_messages):
                    if chunk.content:
                        yield chunk.content

        except Exception as e:
            logger.error("处理问题失败: %s", e, exc_info=True)
            yield f"错误: {e}"


engine = RAGEngine()


# ============ Gradio 界面 ============

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg: #f5f3f0;
  --surface: #ffffff;
  --surface-hover: #faf9f7;
  --primary: #6d28d9;
  --primary-soft: rgba(109,40,217,0.06);
  --primary-mid: rgba(109,40,217,0.12);
  --primary-glow: rgba(109,40,217,0.15);
  --text: #18181b;
  --text-sec: #71717a;
  --text-ter: #a1a1aa;
  --border: #e7e5e4;
  --border-light: #f0efed;
  --radius: 10px;
  --radius-sm: 7px;
  --radius-bubble: 16px;
  --font-serif: 'Playfair Display', Georgia, serif;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

* { font-family: var(--font-sans) !important; }
body, .gradio-container { background: var(--bg) !important; }
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  height: 100dvh !important;
  margin: 0 !important;
  padding: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
}
header { display: none !important; }

.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 24px; background: var(--surface);
  border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.top-bar-left { display: flex; align-items: center; gap: 10px; }
.top-bar h1 {
  font-family: var(--font-serif) !important;
  font-size: 18px; font-weight: 600; color: var(--text);
  letter-spacing: -0.3px; margin: 0;
}
.top-bar .badge {
  font-size: 10px; font-weight: 500; color: var(--primary);
  background: var(--primary-soft); padding: 2px 8px;
  border-radius: 4px; letter-spacing: 0.3px;
}
.top-bar p { font-size: 12px; color: var(--text-ter); margin: 0; font-weight: 400; }

#main-layout {
  display: flex !important; flex-direction: row !important;
  flex-wrap: nowrap !important; align-items: stretch !important;
  gap: 0 !important; flex: 1 !important; min-height: 0 !important; padding: 0 !important;
}
#main-layout > div:first-child {
  width: 270px !important; min-width: 270px !important; max-width: 270px !important;
  flex: 0 0 270px !important; padding: 16px !important;
  display: flex !important; flex-direction: column !important; gap: 10px !important;
  border-right: 1px solid var(--border) !important;
  background: var(--surface) !important; overflow-y: auto !important;
}
#main-layout > div:last-child {
  flex: 1 !important; min-width: 0 !important; min-height: 0 !important;
  height: 100% !important; display: flex !important; flex-direction: column !important;
  background: var(--surface) !important;
  padding-bottom: max(20px, env(safe-area-inset-bottom, 20px)) !important;
}

.sidebar-section { background: transparent; border-radius: var(--radius-sm); padding: 0; }
.sidebar-section .section-label {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.8px; color: var(--text-ter); margin-bottom: 10px;
  display: flex; align-items: center; gap: 5px;
}

.gr-file {
  border: 2px dashed var(--border) !important; border-radius: var(--radius-sm) !important;
  background: var(--surface-hover) !important; padding: 24px 12px !important;
  transition: all .2s ease !important; cursor: pointer !important;
  min-height: 80px !important; display: flex !important;
  align-items: center !important; justify-content: center !important;
}
.gr-file:hover { border-color: var(--primary) !important; background: var(--primary-soft) !important; border-width: 2px !important; }
.gr-file:focus-within { border-color: var(--primary) !important; box-shadow: 0 0 0 3px var(--primary-glow) !important; }
.gr-file .file-preview { border: none !important; padding: 0 !important; }
.file-preview-div { background: transparent !important; border: none !important; }
.gr-file, .gr-file .file-preview, .gr-file .file-preview-div { min-height: 80px !important; display: flex !important; align-items: center !important; justify-content: center !important; }
.gr-form:has(> .gr-file) { min-height: 80px !important; }
.upload-hint { font-size: 12px; color: var(--text-ter); text-align: center; margin-top: -12px; margin-bottom: 2px; }

.gr-text-input, .gr-text-input input {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border-light) !important;
  background: var(--surface-hover) !important;
  font-size: 12.5px !important; color: var(--text) !important;
  padding: 8px 10px !important; transition: border-color .15s ease !important;
}
.gr-text-input:has(#status-display) {
  background: var(--primary-soft) !important; border-color: var(--primary-mid) !important;
  max-height: 40px !important; overflow: hidden !important;
}
.gr-text-input:has(#status-display) input {
  background: transparent !important; border-color: transparent !important;
  color: var(--primary) !important; font-weight: 500 !important;
  cursor: default !important; white-space: nowrap !important;
  overflow: hidden !important; text-overflow: ellipsis !important;
}

.chat-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px 10px 20px;
  border-bottom: 1px solid var(--border-light);
  flex-shrink: 0; background: var(--surface);
}
.chat-header-label {
  font-size: 12px; font-weight: 500; color: var(--text-sec);
  text-transform: uppercase; letter-spacing: 0.5px;
  display: flex; align-items: center; gap: 5px;
}
.chat-header-meta { font-size: 11px; color: var(--text-ter); font-weight: 400; }

.thinking-dots {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 24px 4px 24px; flex-shrink: 0;
}
.thinking-dots .dots { display: flex; gap: 4px; align-items: center; }
.thinking-dots .dots span {
  width: 6px; height: 6px; border-radius: 50%; background: var(--text-ter);
  animation: dotBounce 1.4s ease-in-out infinite;
}
.thinking-dots .dots span:nth-child(1) { animation-delay: 0s; }
.thinking-dots .dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots .dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dotBounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.35; }
  30% { transform: translateY(-6px); opacity: 1; }
}
.thinking-dots .label { font-size: 11.5px; color: var(--text-ter); font-weight: 400; }

.chatbot-area {
  flex: 1 1 1px !important; min-height: 0 !important; height: 100% !important;
  max-height: 100% !important; display: flex !important;
  flex-direction: column !important; overflow: hidden !important;
}
.chatbot-area .chatbot { flex: 1 !important; min-height: 0 !important; height: 100% !important; border: none !important; background: transparent !important; }
.chatbot-area > div { min-height: 0 !important; height: 100% !important; display: flex !important; flex-direction: column !important; }
.chatbot .wrap { overflow-y: auto !important; height: 100% !important; flex: 1 !important; }

.bubble-wrap .message-wrap { display: flex !important; align-items: flex-end !important; gap: 8px !important; margin: 6px 0 !important; padding: 0 16px !important; }
.bubble-wrap .message-wrap.user { flex-direction: row-reverse !important; justify-content: flex-start !important; }
.bubble-wrap .message-wrap.assistant { flex-direction: row !important; justify-content: flex-start !important; }

@keyframes messageSlideUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.bubble-wrap .message-wrap:last-child { animation: messageSlideUp 0.35s ease-out !important; }

.user-message, .assistant-message { font-size: 14px !important; line-height: 1.65 !important; padding: 10px 16px !important; max-width: 75% !important; }
.user-message { background: var(--primary) !important; color: #fff !important; border-radius: var(--radius-bubble) var(--radius-bubble) 4px var(--radius-bubble) !important; margin: 0 !important; font-weight: 400 !important; }
.assistant-message { background: var(--surface-hover) !important; color: var(--text) !important; border: 1px solid var(--border-light) !important; border-radius: var(--radius-bubble) var(--radius-bubble) var(--radius-bubble) 4px !important; margin: 0 !important; }
.assistant-message p { margin: 0 0 6px 0; }
.assistant-message p:last-child { margin-bottom: 0; }

[class*="chatbot-area"] button { display: none !important; }

.chat-divider { border-top: 1px solid var(--border-light); margin: 0 16px; flex-shrink: 0; }

#chat-input-area { flex-shrink: 0 !important; padding-bottom: max(0px, env(safe-area-inset-bottom, 36px)) !important; }
#chat-input-area .gr-row:first-of-type { padding: 12px 16px 4px !important; }
#chat-input-area .gr-row:last-of-type { padding: 0 16px 10px !important; }

textarea {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  background: var(--surface-hover) !important;
  padding: 10px 14px !important; font-size: 14px !important;
  color: var(--text) !important; resize: none !important;
  transition: all .2s ease !important; line-height: 1.5 !important;
  min-height: 42px !important;
}
textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--primary-glow) !important;
  outline: none !important; background: var(--surface) !important;
}
textarea::placeholder { color: var(--text-ter) !important; }

button, .gr-button {
  border: none !important; border-radius: var(--radius-sm) !important;
  padding: 9px 16px !important; font-size: 13px !important;
  font-weight: 500 !important; cursor: pointer !important;
  transition: all .2s ease !important; letter-spacing: 0 !important;
}
button.primary, .gr-button-primary { background: var(--primary) !important; color: #fff !important; }
button.primary:hover { background: #5b21b6 !important; box-shadow: 0 2px 10px var(--primary-glow) !important; }
button.primary:active { transform: scale(0.98); }
button.secondary, .gr-button-secondary { background: transparent !important; color: var(--text-sec) !important; border: 1px solid var(--border) !important; }
button.secondary:hover { background: var(--surface-hover) !important; color: var(--text) !important; }

.gr-box, .input-container, .wrap, .panel, .tab-nav, .form,
.gr-form, .block, .container, .gr-group { border: none !important; box-shadow: none !important; background: transparent !important; }
label, .block-label, .label-text { font-size: 13px !important; font-weight: 500 !important; color: var(--text) !important; margin-bottom: 3px !important; }
footer { display: none !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-ter); }
.gr-row { gap: 6px !important; }
.gr-column { gap: 6px !important; }
#send-btn { min-width: 70px !important; }
"""

import gradio as gr


def initialize():
    engine.vectorstore = None
    engine.update_tools()
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
            gr.HTML("""
            <script>
            (function() {
                function patch() {
                    var wrap = document.querySelector('[class*="chatbot"] [class*="wrap"]');
                    if (!wrap) return;
                    wrap.scrollTop = wrap.scrollHeight;
                    var children = wrap.children;
                    for (var i = 0; i < children.length; i++) {
                        var row = children[i];
                        if (row.querySelector('.av') || row.children.length === 0) continue;
                        var txt = (row.textContent || '').trim();
                        if (txt.length < 3) continue;
                        var isUser = /user/i.test((row.className || '') + ' ' + (row.getAttribute('class') || ''));
                        var av = document.createElement('span');
                        av.className = 'av';
                        av.textContent = isUser ? 'U' : 'AI';
                        av.style.cssText = 'display:inline-flex;align-items:center;justify-content:center;width:30px;min-width:30px;height:30px;border-radius:8px;font-size:11px;font-weight:600;flex-shrink:0;margin:0;' + (isUser ? 'background:#6d28d9;color:#fff;' : 'background:#faf9f7;color:#71717a;border:1px solid #e7e5e4;');
                        row.insertBefore(av, row.firstChild);
                    }
                }
                var oldLen = 0;
                setInterval(function() {
                    var wrap = document.querySelector('[class*="chatbot"] [class*="wrap"]');
                    if (!wrap) return;
                    var cur = wrap.children.length;
                    if (cur !== oldLen || !wrap.querySelector('.av')) { oldLen = cur; patch(); }
                }, 300);
            })();
            </script>
            """)
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
            new_history = list(chat_history) + [{"role": "user", "content": message}]
            partial = ""
            for chunk in engine.respond_stream(message, chat_history):
                partial += chunk
                yield "", new_history + [{"role": "assistant", "content": partial}]
            logger.info("助手回复完成, 长度: %d", len(partial))
        except Exception as e:
            logger.error("respond_wrapper 异常: %s", e, exc_info=True)
            yield "", chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"系统错误: {e}"},
            ]

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


if __name__ == "__main__":
    print("\n  -> 访问地址: http://127.0.0.1:7860\n")
    demo.launch(inbrowser=False, quiet=True, show_error=True)
