import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from tests import *  # noqa: F401, F403
from src.rag_engine import _table_to_markdown


class TestTableToMarkdown:
    def test_basic_table(self):
        data = [["姓名", "年龄"], ["张三", "28"], ["李四", "35"]]
        result = _table_to_markdown(data)
        expected = (
            "| 姓名 | 年龄 |\n"
            "| --- | --- |\n"
            "| 张三 | 28 |\n"
            "| 李四 | 35 |"
        )
        assert result == expected

    def test_single_row(self):
        """只有表头的情况"""
        data = [["产品", "价格"]]
        result = _table_to_markdown(data)
        expected = "| 产品 | 价格 |\n| --- | --- |"
        assert result == expected

    def test_empty_data(self):
        assert _table_to_markdown([]) == ""
        assert _table_to_markdown([[]]) == ""

    def test_uneven_rows(self):
        """行长度不一致时补齐或截断"""
        data = [["A", "B", "C"], ["short"], ["too", "many", "cols", "here"]]
        result = _table_to_markdown(data)
        lines = result.split("\n")
        # Markdown 表格格式：| A | B | C |，split("|") 得到 ['', ' A ', ' B ', ' C ', '']
        # 3 列数据 = 4 个 |，split 后应为 5 个元素
        assert len(lines[0].split("|")) == 5  # ['', ' A ', ' B ', ' C ', '']
        # 第二行补齐到 3 列
        assert len(lines[2].split("|")) == 5
        # 第三行截断到 3 列
        assert len(lines[3].split("|")) == 5

    def test_newlines_in_cells(self):
        """单元格内的换行符替换为空格"""
        data = [["名称"], ["多行\n文本"]]
        result = _table_to_markdown(data)
        assert "多行 " in result
        assert "\n" not in result.split("|")[1]

    def test_numeric_data_preserved(self):
        data = [["季度", "销售额"], ["Q1", "1,280"], ["Q2", "856.5"]]
        result = _table_to_markdown(data)
        assert "1,280" in result
        assert "856.5" in result
