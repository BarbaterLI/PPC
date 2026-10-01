"""扩展可依赖的稳定 CLI 渲染 API —— 宿主与插件之间的公共接缝。

插件/扩展的 CLI 实现不应直接导入 ``src.cli.design.tokens``、``src.cli.output``
等宿主内部模块（宿主内部重构会直接破坏插件）。本模块是面向扩展的公共契约，
承诺以下符号在 12.x 内保持兼容：

* ``console``     —— 与宿主一致的 Rich ``Console`` 实例
* ``ColorTokens`` —— 语义颜色令牌（success / warning / error / info 等）
* ``get_icon``    —— 图标解析（自动遵循 ``--no-emoji`` 与平台能力）

插件请使用::

    from src.cli.extension_api import ColorTokens, console, get_icon

新增导出前必须确认其可长期稳定；内部实现细节不得进入本模块。
"""

from src.cli.design.tokens import ColorTokens, get_icon
from src.cli.output import console

__all__ = ["ColorTokens", "console", "get_icon"]