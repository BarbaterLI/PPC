"""Config store protocol and signature helpers.

manager_core 与 manager_io 原先互相导入形成循环:manager_io 反向依赖
manager_core 仅为了签名 helper。将 helper 与 ``ConfigStoreProtocol`` 协议
下沉到本叶子模块后,manager_core 可安全地在顶层导入 manager_io,循环消除。

``ConfigStoreProtocol`` 是 config IO 函数所依赖的 manager 对象最小面;
具体内存状态容器见 :mod:`src.config.manager_state` 的 ``ConfigStore``。
"""

import hashlib
import hmac
import os
import platform
import sys
from pathlib import Path
from typing import Any, Protocol

from src.config.schema import PPCConfig

_CONFIG_HMAC_KEY_ENV = "PPC_CONFIG_SIGN_KEY"
_CONFIG_SIGNATURE_MARKER = "# __ppc_signature__: "


class ConfigStoreProtocol(Protocol):
    """config IO 函数所依赖的 manager 对象最小面(ConfigManager 结构性满足)。"""

    CONFIG_VERSION: str
    config_path: Path
    ppc5_config_path: Path
    _config: PPCConfig | None
    _config_dict: dict[str, Any]
    _temp_overrides: dict[str, Any]
    _dirty: bool


def _get_sign_key() -> bytes:
    key = os.environ.get(_CONFIG_HMAC_KEY_ENV, "")
    if key:
        return key.encode("utf-8")
    machine_id = platform.node() + sys.executable
    return hashlib.sha256(machine_id.encode("utf-8")).digest()


def _compute_config_signature(content: str) -> str:
    return hmac.new(_get_sign_key(), content.encode("utf-8"), hashlib.sha256).hexdigest()  # type: ignore[no-any-return]
