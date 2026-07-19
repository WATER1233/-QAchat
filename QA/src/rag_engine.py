from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
import logging
import src.config as cfg
from src.embeddings import AliyunEmbeddings

logger = logging.getLogger(__name__)

NEED_SEARCH_PROMPT = """你是一个问答路由。严格按以下规则判断用户问题是否需要查阅文档。

规则：
- "yes" → 问题是关于文档内容、文档分析、文档查询、摘要等具体信息
- "no"  → 日常闲聊、问候、感谢、告别、通用常识、询问 AI 本身、无关话题

示例：
  问：报告中的营收数据是多少 → yes
  问：文档第三章讲了什么 → yes
  问：帮我总结一下这份 PDF → yes
  问：你好                  → no
  问：今天天气怎么样         → no
  问：谢谢                  → no
  问：你是谁                → no
  问：1+1等于几             → no
  问：你能做什么             → no
  问：没事了                 → no

只输出 yes 或 no，不要其它内容。

问题：{query}
结果："""


def _no_search_keyword_match(q: str) -> bool:
    """关键词预匹配，命中直接返回 False（不查文档），覆盖大部分闲聊场景。"""
    keywords = [
        # 问候
        "你好", "hello", "hi", "嗨", "嗨喽", "哈喽", "hey",
        "早上好", "下午好", "晚上好", "晚安",
        # 询问 AI 身份/能力
        "你是谁", "你叫什么", "你叫什么名字", "你是什么",
        "你能做什么", "你会什么", "你有什么功能",
        # 礼貌用语
        "谢谢", "感谢", "多谢", "辛苦了", "谢谢你",
        "拜拜", "再见", "88", "bye",
        # 日常闲聊
        "在干嘛", "在吗", "在不在", "干嘛呢",
        "晚上吃什么", "中午吃什么", "今天吃什么",
        "今天天气", "明天天气",
        "没事", "没事了", "没什么", "算了", "好吧",
        # 通用知识（不需要查文档）
        "1+1", "2+2", "等于几", "等于多少",
        "现在几点", "几点了", "今天几号",
    ]
    for kw in keywords:
        if kw in q:
            return True
    return False


def need_search(llm, query: str) -> bool:
    """判断是否需要检索文档：关键词预匹配 → LLM 路由兜底。"""
    q = query.strip().lower()
    if _no_search_keyword_match(q):
        return False
    try:
        resp = llm.invoke([
            SystemMessage(content=NEED_SEARCH_PROMPT.format(query=query)),
        ]).content.strip().lower()
        return resp == "yes"
    except Exception:
        return True

NO_RESULT = "未检索到相关内容"

try:
    import fitz
    HAS_PYMUPDF = True
    logger.info("使用 pymupdf 解析 PDF")
except ImportError:
    HAS_PYMUPDF = False
    logger.info("pymupdf 未安装，使用 pypdf 解析 PDF")


def extract_pdf_pages(pdf_path: str):
    """提取 PDF 所有页的文本，优先用 pymupdf（更好的表格/多栏处理），回退到 pypdf。"""
    pages = []
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages.append(Document(page_content=text.strip(), metadata={"page": i + 1}))
            doc.close()
            if pages:
                return pages
        except Exception as e:
            logger.warning("pymupdf 解析失败: %s，回退到 pypdf", e)

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append(Document(page_content=text.strip(), metadata={"page": i + 1}))
    return pages


def build_vectorstore(llm, pdf_files, existing_vectorstore=None):
    """pdf_files: list of (file_path, display_name)
       existing_vectorstore: 已有 Chroma 实例时增量追加，否则新建
       逐 PDF 处理、逐批入库，避免全量 chunk 驻留内存。"""
    embeddings = AliyunEmbeddings()
    vectorstore = existing_vectorstore
    summaries_text = []

    for pdf_path, pdf_name in pdf_files:
        logger.info("开始建库，PDF: %s (%s)", pdf_name, pdf_path)
        docs = extract_pdf_pages(pdf_path)
        if not docs:
            logger.warning("  %s: 未能提取到文本内容（可能是扫描件）", pdf_name)
            continue
        logger.info("  %s: 已加载 %d 页", pdf_name, len(docs))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source_pdf"] = pdf_name
        logger.info("  %s: 已分割为 %d 个段落", pdf_name, len(chunks))

        # 汇总全文本用于摘要（在释放 docs 之前提取）
        full_text = "\n".join(d.page_content for d in docs)
        # 立即入库，不累计全量 chunk
        if vectorstore:
            vectorstore.add_documents(chunks)
        else:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=cfg.CHROMA_DIR,
            )
        logger.info("  %s: 向量库写入完成", pdf_name)
        # 主动释放引用，允许 GC 回收
        del chunks, docs

        # 文档摘要（少量文本，不影响内存）
        try:
            summary = llm.invoke([
                SystemMessage(content="用一句话简要概括以下文档的核心内容，不超过50字。"),
                HumanMessage(content=full_text[:2000]),
            ]).content.strip()
            summaries_text.append(f"[{pdf_name}] {summary}")
            vectorstore.add_documents([Document(
                page_content=f"【文档摘要】{pdf_name}：{summary}",
                metadata={"source_pdf": pdf_name, "is_summary": True, "page": 0},
            )])
            logger.info("  %s: 摘要生成完成", pdf_name)
        except Exception as e:
            logger.warning("  %s: 摘要生成失败: %s", pdf_name, e)

    if vectorstore is None:
        return existing_vectorstore, summaries_text

    logger.info("向量库已保存")
    return vectorstore, summaries_text


REWRITE_PROMPT = """你是一个搜索关键词优化助手。用户的原始问题可能比较简略，请将其重写为更完整、适合向量检索的关键词或短语。
只输出改写后的文本，不要解释。

原始问题：{query}
改写后："""


def rewrite_query(llm, query: str) -> str:
    try:
        rewritten = llm.invoke([
            SystemMessage(content=REWRITE_PROMPT.format(query=query)),
        ]).content.strip()
        logger.info("查询改写: '%s' → '%s'", query[:50], rewritten[:80])
        return rewritten
    except Exception:
        return query


MULTI_QUERY_PROMPT = """你是一个AI助手，请根据用户的问题，生成最多3个不同角度的检索查询。
这些查询应该从不同维度覆盖同一个主题，以提高文档检索的召回率。
每个查询一行，不要编号，不要多余内容。如果问题很简单，可以少于3个。

用户问题：{query}
检索查询："""

HYDE_PROMPT = """你是一位文档撰写者。根据以下问题，写一段**假设性的文档段落**作为回答。
要求：语言客观、信息密集、结构清晰，像真实的技术文档/报告节选。
只输出段落内容，不要解释。

问题：{query}
假设文档段落："""


def hyde_generate(llm, query: str) -> str | None:
    """HyDE: 生成假设文档段落用于检索"""
    try:
        response = llm.invoke([
            SystemMessage(content=HYDE_PROMPT.format(query=query)),
        ]).content.strip()
        if response:
            logger.info("HyDE 生成完成: %s...", response[:60])
            return response
    except Exception as e:
        logger.warning("HyDE 生成失败: %s", e)
    return None


def rerank_docs(query: str, docs: list[Document], top_n: int = None) -> list[Document]:
    """调用阿里云百炼 Reranker API 对文档精排，返回排序后的 top_n 条"""
    if not docs:
        return docs
    top_n = top_n or cfg.RERANK_TOP_N
    documents = [
        f"[{d.metadata.get('source_pdf', '文档')}·第{d.metadata.get('page', '?')}页] {d.page_content}"
        for d in docs
    ]
    try:
        import requests
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            headers={
                "Authorization": f"Bearer {cfg.API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.RERANK_MODEL,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": top_n},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("Reranker 返回异常: %s %s", resp.status_code, resp.text[:200])
            return docs[:top_n]
        body = resp.json()
        results = body.get("output", {}).get("results", [])
        ranked = []
        for r in results:
            idx = r.get("index")
            if idx is not None and idx < len(docs):
                ranked.append(docs[idx])
        logger.info("Reranker 精排完成: %d -> %d 条", len(docs), len(ranked))
        return ranked
    except Exception as e:
        logger.warning("Reranker 调用失败，使用原始排序: %s", e)
        return docs[:top_n]


def multi_query_retrieve(llm, vectorstore, query: str) -> str | None:
    """Multi-Query 检索 + HyDE：并行执行多角度关键词 + 假设文档段落，去重合并后返回"""
    multi_queries = []
    hyde_text = None

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=2) as _pool:
        _mq = _pool.submit(
            lambda: llm.invoke([
                SystemMessage(content=MULTI_QUERY_PROMPT.format(query=query)),
            ]).content.strip()
        )
        _hy = _pool.submit(hyde_generate, llm, query)
        try:
            response = _mq.result()
            multi_queries = [q.strip() for q in response.split("\n") if q.strip()][:3]
        except Exception:
            pass
        try:
            hyde_text = _hy.result()
        except Exception:
            pass

    queries = [query] + multi_queries
    if hyde_text:
        queries.append(hyde_text)
    logger.info("Multi-Query + HyDE 检索: %d 个查询", len(queries))

    seen = set()
    all_docs = []
    for q in queries:
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(q, k=cfg.RETRIEVAL_K)
        for d, score in docs_with_scores:
            if score < cfg.SIMILARITY_THRESHOLD:
                continue
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    if not all_docs:
        return None

    # Reranker 精排，取 top_n 条
    all_docs = rerank_docs(query, all_docs)

    return "\n\n".join(
        f"[{d.metadata.get('source_pdf', '文档')}·第{d.metadata.get('page', '?')}页] {d.page_content}"
        for d in all_docs
    )


def retrieve(llm, vectorstore, query: str):
    rewritten = rewrite_query(llm, query)
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(rewritten, k=cfg.RETRIEVAL_K)
    docs = [d for d, s in docs_with_scores if s >= cfg.SIMILARITY_THRESHOLD]
    if not docs:
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=cfg.RETRIEVAL_K)
        docs = [d for d, s in docs_with_scores if s >= cfg.SIMILARITY_THRESHOLD]
    if not docs:
        return None

    docs = rerank_docs(query, docs)

    return "\n\n".join(
        f"[{d.metadata.get('source_pdf', '文档')}·第{d.metadata.get('page', '?')}页] {d.page_content}"
        for d in docs
    )


