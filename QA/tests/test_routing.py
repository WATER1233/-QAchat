from unittest.mock import Mock, patch

import pytest

from tests import *  # noqa: F401, F403
from src.rag_engine import need_search, _no_search_keyword_match


class TestNoSearchKeywordMatch:
    """关键词预匹配测试——不依赖 LLM"""

    def test_greeting(self):
        assert _no_search_keyword_match("你好") is True
        assert _no_search_keyword_match("hello") is True
        assert _no_search_keyword_match("早上好") is True

    def test_ai_identity(self):
        assert _no_search_keyword_match("你是谁") is True
        assert _no_search_keyword_match("你能做什么") is True
        assert _no_search_keyword_match("你叫什么名字") is True

    def test_politeness(self):
        assert _no_search_keyword_match("谢谢") is True
        assert _no_search_keyword_match("辛苦了") is True
        assert _no_search_keyword_match("再见") is True

    def test_small_talk(self):
        assert _no_search_keyword_match("在吗") is True
        assert _no_search_keyword_match("没事了") is True
        assert _no_search_keyword_match("今天天气怎么样") is True

    def test_general_knowledge(self):
        assert _no_search_keyword_match("1+1等于几") is True
        assert _no_search_keyword_match("现在几点") is True

    def test_doc_related_not_matched(self):
        """文档相关问题不应被关键词过滤"""
        assert _no_search_keyword_match("报告第三章的营收数据") is False
        assert _no_search_keyword_match("帮我总结一下这份文档") is False
        assert _no_search_keyword_match("实验数据中的平均值是多少") is False

    def test_case_sensitive(self):
        """_no_search_keyword_match 本身大小写敏感"""
        assert _no_search_keyword_match("hello") is True
        assert _no_search_keyword_match("你好") is True
        # 大写不匹配（need_search 会在调用前转小写）
        assert _no_search_keyword_match("HELLO") is False
        assert _no_search_keyword_match("Hi") is False


class TestNeedSearch:
    """need_search 测试——关键词预匹配 + LLM 路由兜底"""

    def test_keyword_filter_returns_false(self):
        """关键词命中直接返回 False"""
        result = need_search(Mock(), "你好")
        assert result is False

    def test_keyword_filter_multiple_spaces(self):
        """带多余空格的输入"""
        result = need_search(Mock(), "  你 好  ")
        assert result is False

    def test_llm_return_yes(self):
        """LLM 返回 yes → 需要检索"""
        llm = Mock()
        llm.invoke.return_value.content = "yes"
        result = need_search(llm, "这份报告里的核心结论是什么")
        assert result is True

    def test_llm_return_no(self):
        """LLM 返回 no → 不需要检索"""
        llm = Mock()
        llm.invoke.return_value.content = "no"
        result = need_search(llm, "今天过得怎么样")
        assert result is False

    def test_llm_exception_default_true(self):
        """LLM 异常时默认返回 True（安全起见）"""
        llm = Mock()
        llm.invoke.side_effect = Exception("API error")
        result = need_search(llm, "随便问问")
        assert result is True

    def test_llm_response_whitespace(self):
        """LLM 返回带空格的响应"""
        llm = Mock()
        llm.invoke.return_value.content = "  yes  "
        result = need_search(llm, "文档里的表格数据")
        assert result is True
