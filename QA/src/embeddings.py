import time
import logging
from langchain_core.embeddings import Embeddings
from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor

import src.config as cfg

logger = logging.getLogger(__name__)

API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class AliyunEmbeddings(Embeddings):
    def __init__(self, model: str = cfg.EMBEDDING_MODEL):
        self.base_url = cfg.BASE_URL.rstrip("/") + "/embeddings"
        self.api_key = cfg.API_KEY
        self.model = model

    def _call_api(self, text: str) -> List[float]:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": text},
                    timeout=API_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning("Embedding API 请求失败 (重试 %d/%d): %s", attempt + 1, MAX_RETRIES, e)
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
            except requests.HTTPError as e:
                if e.response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                    last_error = e
                    logger.warning("Embedding API 服务端错误 (重试 %d/%d): %s", attempt + 1, MAX_RETRIES, e)
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"Embedding API 请求重试 {MAX_RETRIES} 次后仍失败: {last_error}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float] | None] = [None] * len(texts)
        batch_size = cfg.EMBEDDING_BATCH_SIZE

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            logger.debug("Embedding 批次 [%d, %d) / %d", start, start + len(batch), len(texts))

            with ThreadPoolExecutor(max_workers=cfg.EMBEDDING_MAX_WORKERS) as pool:
                futures = {i: pool.submit(self._call_api, t) for i, t in enumerate(batch)}
                for local_i, global_i in enumerate(range(start, start + len(batch))):
                    results[global_i] = futures[local_i].result()

        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        return self._call_api(text)
