"""reliability 包导出面（``__all__``）一致性测试。"""

from __future__ import annotations

import src.reliability as rel


def test_all_entries_are_defined():
    """``__all__`` 中的每个名字都必须真实存在。

    Bug: ``__all__`` 列了 ``Ok`` / ``Err`` / ``is_ok`` / ``is_err`` 四个从未
    实现的名字（死条目），导致 ``from src.reliability import *`` 对每个名字做
    ``getattr`` 时抛 ``AttributeError: module ... has no attribute 'Ok'``，
    通配导入完全不可用。
    """
    missing = [n for n in rel.__all__ if not hasattr(rel, n)]
    assert missing == [], f"__all__ 引用了未定义的名字: {missing}"


def test_wildcard_import_works():
    """``from src.reliability import *`` 不得抛 AttributeError。"""
    ns: dict = {}
    exec("from src.reliability import *", ns)  # noqa: S102 - 验证通配导入
    exported = [k for k in ns if not k.startswith("__")]
    assert len(exported) == len(rel.__all__)
