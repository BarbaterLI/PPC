"""PPC_NEXT 版本号单一来源 / Single source of truth for the PPC_NEXT version.

所有模块（``ppc.py``、``src/cli/output.py``、``src/web/api/system.py``、
``src/extensions/package.py`` 等）都应从本模块读取 ``__version__``，
避免版本号散落多处、以及 ``from ppc import __version__`` 在
``python ppc.py`` 启动方式下触发 ppc 模块二次导入。
"""

__version__ = "12.0.1-rc2-pre.1"

__all__ = ["__version__"]
