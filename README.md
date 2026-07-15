# PDF 智能问答系统

基于 RAG（Retrieval-Augmented Generation）的 PDF 文档智能问答系统。上传 PDF 文档后，通过自然语言对话方式查询文档内容，支持多文档统一检索和来源标注。

## 功能特性

- **多文档统一检索**：同时上传多个 PDF，自动建库，跨文档统一检索
- **Multi-Query 检索**：一次提问生成多个角度查询词分别检索、去重合并，提升召回率
- **对话上下文缓存**：追问时自动带入上次检索结果，无需重复搜索
- **智能查询改写**：LLM 自动优化用户查询，提高检索命中率
- **Tool Calling 判断**：LLM 自主判断是否触发文档检索，闲聊不干扰
- **流式输出**：SSE 流式响应，实时展示回答生成过程
- **来源标注**：回答自动标注来源文档和页码，方便核对原文
- **双模式切换**：无文档时自动降级为普通聊天模式

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 UI | Gradio |
| RAG 框架 | LangChain |
| 向量库 | ChromaDB |
| Embedding | text-embedding-v3（阿里云百炼） |
| LLM | Qwen-Flash（阿里云百炼） |
| PDF 解析 | pypdf |

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入阿里云百炼的 API 密钥和 Endpoint：

```
BASE_URL=https://your-region.aliyuncs.com/compatible-mode/v1
API_KEY=sk-xxxxxxxxxxxxxxxx
MODEL_NAME=qwen-flash
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动

```bash
python app.py
```

访问 http://127.0.0.1:7860

## 项目结构

```
QA/
├── app.py                 # Gradio UI 入口
├── src/
│   ├── __init__.py
│   ├── config.py           # 配置加载
│   ├── embeddings.py       # 阿里云 Embedding 封装
│   └── rag_engine.py       # RAG 核心逻辑
├── .env                    # 本地配置（不提交）
├── .env.example            # 配置模板
├── requirements.txt
├── .gitignore
└── README.md
```

## 检索流程

```
用户提问
  ├─ 有文档 → LLM 判断是否调 SearchPDF 工具
  │            ├─ 是 → Multi-Query 多角度检索 → 去重合并 → LLM 生成回答 ✅
  │            │                                          ↘ 缓存上下文，支持追问
  │            └─ 否 → 有缓存 → 注入上次上下文 → 普通聊天（带文档背景）
  │                  → 无缓存 → 普通聊天
  └─ 无文档 → 普通聊天
```

## License

MIT
