# ===== 阶段 1：构建前端 =====
FROM node:20-alpine AS frontend-builder

WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ===== 阶段 2：运行后端 =====
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（pymupdf 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY api.py .
COPY src/ src/
COPY --from=frontend-builder /build/dist/ frontend/dist/

# 运行时数据目录
VOLUME ["/app/chroma_db"]

EXPOSE 7860

CMD ["python", "api.py"]
