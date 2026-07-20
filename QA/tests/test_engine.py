from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict

import pytest

from tests import *  # noqa: F401, F403
from api import RAGEngine, CHAT_SYSTEM_PROMPT, QA_SYSTEM_PROMPT, LRUCache


@pytest.fixture
def engine():
    """创建一个 mock 所有外部依赖的 RAGEngine"""
    eng = RAGEngine()
    eng.llm = Mock()
    # 使用真实 LRUCache（Mock 对象不支持内部 _data 操作）
    eng._answer_cache = LRUCache(maxsize=100, ttl=300)
    return eng


def _mock_stream(content: str):
    """模拟 LLM 流式返回"""
    class Chunk:
        def __init__(self, text):
            self.content = text
    for ch in content:
        yield Chunk(ch)


class TestRAGEngineStream:
    """respond_stream 的四条路径"""

    def test_no_vectorstore_chat(self, engine):
        """路径 1：无文档 → 纯聊天"""
        engine.vectorstore = None
        engine.llm.stream.return_value = _mock_stream("你好呀！")

        tokens = list(engine.respond_stream("你好", []))
        assert len(tokens) > 0
        full = "".join(tokens)
        assert "你好" in full or "你好呀" in full

        # 验证使用了 CHAT_SYSTEM_PROMPT
        call_kwargs = engine.llm.stream.call_args[0][0]
        system_msg = call_kwargs[0]
        assert system_msg.content == CHAT_SYSTEM_PROMPT

    def test_cache_hit_exact(self, engine):
        """路径 2：缓存命中 → 直接返回"""
        engine.vectorstore = Mock()
        # 往真实缓存写入数据
        engine._answer_cache.put("重复问题", "缓存的回答")

        with patch.object(engine, '_get_cached_answer', wraps=engine._get_cached_answer):
            tokens = list(engine.respond_stream("重复问题", []))
            assert "".join(tokens) == "缓存的回答"

    def test_with_document_retrieve_hit(self, engine):
        """路径 3：有文档且检索命中"""
        engine.vectorstore = MagicMock()

        with (
            patch("api.need_search", return_value=True),
            patch("api.multi_query_retrieve", return_value="检索到的资料内容"),
        ):
            engine.llm.stream.return_value = _mock_stream("根据资料，答案是42。")

            tokens = list(engine.respond_stream("答案是什么", []))
            full = "".join(tokens)
            assert len(full) > 0

            # 验证使用了 QA_SYSTEM_PROMPT（含检索到的资料）
            call_kwargs = engine.llm.stream.call_args[0][0]
            system_msg = call_kwargs[0]
            assert "检索到的资料" in system_msg.content
            # verify no_search returned False so we used search path
            assert engine.vectorstore is not None

    def test_no_search_no_last_context(self, engine):
        """路径 4a：无需检索文档，无缓存上下文 → 纯聊天"""
        engine.vectorstore = Mock()
        engine._last_context = None

        with patch("api.need_search", return_value=False):
            engine.llm.stream.return_value = _mock_stream("随意聊聊~")

            tokens = list(engine.respond_stream("今天天气", []))
            full = "".join(tokens)
            assert len(full) > 0

            # 验证用了 CHAT_SYSTEM_PROMPT
            call_kwargs = engine.llm.stream.call_args[0][0]
            assert call_kwargs[0].content == CHAT_SYSTEM_PROMPT

    def test_no_search_with_last_context(self, engine):
        """路径 4b：无需检索，但有上次检索上下文 → 注入上下文"""
        engine.vectorstore = Mock()
        engine._last_context = "上次检索到的文档内容"

        with patch("api.need_search", return_value=False):
            engine.llm.stream.return_value = _mock_stream("基于上次的资料回答。")

            tokens = list(engine.respond_stream("继续说说", []))
            full = "".join(tokens)
            assert len(full) > 0

            # 验证使用了 QA_SYSTEM_PROMPT 且包含了上次的上下文
            call_kwargs = engine.llm.stream.call_args[0][0]
            system_content = call_kwargs[0].content
            assert "上次检索到的文档内容" in system_content

    def test_unexpected_exception(self, engine):
        """路径 5：检索异常 → 返回错误信息"""
        engine.vectorstore = Mock()

        with patch("api.need_search", return_value=True):
            with patch("api.multi_query_retrieve", side_effect=Exception("API挂了")):
                tokens = list(engine.respond_stream("问个问题", []))
                full = "".join(tokens)
                assert "错误" in full or "API" in full


class TestRAGEngineBuildHistory:
    def test_dict_history(self, engine):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀"},
        ]
        messages = engine._build_history(history)
        assert len(messages) == 2
        assert messages[0].type == "human"
        assert messages[1].type == "ai"

    def test_empty_history(self, engine):
        assert engine._build_history([]) == []



class TestRAGEngineCache:
    def test_get_cached_answer_exact(self, engine):
        engine._answer_cache.put("你好", "cached")
        result = engine._get_cached_answer("你好")
        assert result == "cached"
