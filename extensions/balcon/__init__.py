"""balcon TTS 后端扩展 —— Balabolka 命令行工具（Windows SAPI 5）接入 ``ppc.tts.v1``。

扩展包唯一入口：本模块即扩展包根（``__init__.py``），``manifest.yml`` 的
``entry_point: "extension:BalconExtension"`` 中的 ``extension`` 为宿主约定
关键字，指代**当前已加载模块**，因此本模块必须导出同名类。
"""

from .extension import BalconExtension

extension = BalconExtension()

__all__ = ["BalconExtension", "extension"]
