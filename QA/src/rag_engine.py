from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
import shutil
import time
import logging
import os

import src.config as cfg
from src.embeddings import AliyunEmbeddings

logger = logging.getLogger(__name__)


def build_vectorstore(llm, pdf_files):
    """pdf_files: list of (file_path, display_name)"""
    embeddings = AliyunEmbeddings()
    all_chunks = []
    summaries_text = []

    for pdf_path, pdf_name in pdf_files:
        logger.info("开始建库，PDF: %s (%s)", pdf_name, pdf_path)
        reader = PdfReader(pdf_path)
        docs = []
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": len(docs) + 1}))
        logger.info("  %s: 已加载 %d 页", pdf_name, len(docs))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source_pdf"] = pdf_name
        all_chunks.extend(chunks)
        logger.info("  %s: 已分割为 %d 个段落", pdf_name, len(chunks))

        full_text = "\n".join(d.page_content for d in docs)
        try:
            summary = llm.invoke([
                SystemMessage(content="用一句话简要概括以下文档的核心内容，不超过50字。"),
                HumanMessage(content=full_text[:2000]),
            ]).content.strip()
            summaries_text.append(f"[{pdf_name}] {summary}")
            all_chunks.append(Document(
                page_content=f"【文档摘要】{pdf_name}：{summary}",
                metadata={"source_pdf": pdf_name, "is_summary": True, "page": 0},
            ))
            logger.info("  %s: 摘要生成完成", pdf_name)
        except Exception as e:
            logger.warning("  %s: 摘要生成失败: %s", pdf_name, e)

    logger.info("总计 %d 个段落，生成向量中...", len(all_chunks))
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=cfg.CHROMA_DIR,
    )
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


def is_doc_question(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in cfg.DOC_QUESTION_KEYWORDS)


def retrieve(llm, vectorstore, query: str):
    rewritten = rewrite_query(llm, query)
    docs = vectorstore.similarity_search(rewritten, k=cfg.RETRIEVAL_K)
    if not docs:
        docs = vectorstore.similarity_search(query, k=cfg.RETRIEVAL_K)
    if not docs:
        return None
    return "\n\n".join(
        f"[{d.metadata.get('source_pdf', '文档')}·第{d.metadata.get('page', '?')}页] {d.page_content}"
        for d in docs
    )


def remove_chroma_db(path: str):
    if not os.path.exists(path):
        return
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            logger.info("旧向量库已删除")
            return
        except PermissionError:
            if attempt < 4:
                logger.warning("文件被占用，等待重试 (%d/5)...", attempt + 1)
                time.sleep(1)
            else:
                logger.error("删除向量库失败，文件仍被占用")


def make_search_tool(llm, vectorstore):
    """Create a SearchPDF tool bound to the current vectorstore."""

    @tool
    def SearchPDF(query: str) -> str:
        """从已加载的PDF文档中检索相关内容。当用户询问文档中的信息、内容、总结、具体主题时，调用此工具"""
        if vectorstore is None:
            return "未检索到相关内容"
        result = retrieve(llm, vectorstore, query)
        return result if result else "未检索到相关内容"

    return SearchPDF
