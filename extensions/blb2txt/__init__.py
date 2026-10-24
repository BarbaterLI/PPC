"""blb2txt 扩展包。

把 Balabolka 文本抽取工具（``blb2txt.exe``，支持 34 种电子书/文档格式）
接入 PPC_NEXT：对外提供 ``ppc ext blb2txt ...`` 独立命令，定位为宿主
``ppc split`` 的**输入格式增强器** —— 宿主只吃 ``.txt``，本扩展让
EPUB / MOBI / AZW3 / FB2 / DOCX / PDF 等直接进入抽取与分章流程。
"""

from .extension import Blb2TxtExtension

extension = Blb2TxtExtension()

__all__ = ["Blb2TxtExtension", "extension"]
