# PDF 智能问答系统

基于 RAG（Retrieval-Augmented Generation）的 PDF 文档智能问答系统。上传 PDF 后通过自然语言对话查询内容，支持多文档统一检索、表格提取、来源标注。

## 功能特性

- **多文档统一检索**：同时上传多个 PDF，自动建库，跨文档统一检索
- **表格提取**：自动检测 PDF 表格并转为 Markdown 格式入库，结构化数据可检索
- **Multi-Query + HyDE 检索**：多角度查询词扩展 + 假设文档段落，提升召回率
- **Reranker 精排**：召回后交叉编码精排，保留 top-N 最相关段落
- **智能路由**：关键词预筛 + LLM 判断，闲聊不触发文档检索
- **短期记忆**：会话级历史记忆，刷新页面不丢失对话
- **流式输出**：SSE 流式响应，实时展示回答生成过程
- **来源标注**：回答自动标注来源文档和页码，表格来源带 📊 图标
- **双模式切换**：无文档时自动降级为普通聊天模式

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 UI | React 18 + TypeScript + Vite + Tailwind CSS |
| 后端 API | FastAPI (SSE 流式) |
| RAG 框架 | LangChain |
| 向量库 | ChromaDB |
| Embedding | text-embedding-v3（阿里云百炼） |
| LLM | Qwen-Flash（阿里云百炼） |
| Reranker | gte-rerank-v2（阿里云百炼） |
| PDF 解析 | pymupdf（表格检测）/ pypdf（降级） |
| 测试 | pytest（36 个用例） |

## 快速开始

### 1. 配置环境变量

```bash
cp QA/.env.example QA/.env
```

编辑 `QA/.env`，填入阿里云百炼的 API 密钥。

### 2. 安装依赖

```bash
cd QA
pip install -r requirements.txt
```

### 3. 构建前端

```bash
cd frontend
npm install
npm run build
```

### 4. 启动

```bash
cd .. && python api.py
```

访问 **http://127.0.0.1:7860**

## 项目结构

```
QA/
├── api.py                 # FastAPI 入口（前后端一体）
├── frontend/              # React 前端
│   ├── src/
│   │   ├── components/    # UI 组件
│   │   ├── hooks/         # React Hooks
│   │   ├── services/      # API 调用
│   │   └── types/         # 类型定义
│   └── dist/              # 构建产物（供 api.py 挂载）
├── src/
│   ├── config.py          # 配置加载
│   ├── embeddings.py      # Embedding API 封装
│   └── rag_engine.py      # RAG 核心逻辑
├── tests/                 # 测试
│   ├── test_cache.py
│   ├── test_engine.py
│   ├── test_pdf.py
│   └── test_routing.py
├── .env
└── requirements.txt
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | SSE 流式聊天（支持 session_id） |
| `/api/chat/new` | POST | 创建新会话 |
| `/api/chat/{session_id}` | GET | 获取会话历史 |
| `/api/documents/upload` | POST | 上传 PDF |
| `/api/documents/clear` | POST | 清空文档索引 |
| `/api/health` | GET | 健康检查 |

## 检索流程

```
用户提问
  ├─ 有文档 → LLM 判断是否调检索
  │            ├─ 是 → Multi-Query + HyDE 检索 → Reranker 精排
  │            │                                     → LLM 生成回答 ✅
  │            │                                     ↘ 缓存上下文，支持追问
  │            └─ 否 → 有缓存 → 注入上次上下文 → 普通聊天
  │                  → 无缓存 → 普通聊天
  └─ 无文档 → 普通聊天
```

## License

MIT
