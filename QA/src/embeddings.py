from langchain_core.embeddings import Embeddings
from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import src.config as cfg


class AliyunEmbeddings(Embeddings):
    def __init__(self, model: str = cfg.EMBEDDING_MODEL):
        self.base_url = cfg.BASE_URL.rstrip("/") + "/embeddings"
        self.api_key = cfg.API_KEY
        self.model = model

    def _call_api(self, text: str) -> List[float]:
        resp = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._call_api, t) for t in texts]
            results = []
            for f in as_completed(futures):
                results.append(f.result())
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._call_api(text)
