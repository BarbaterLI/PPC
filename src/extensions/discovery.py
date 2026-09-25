"""扩展发现与加载顺序解析。

* :func:`discover_extensions` — 扫描扩展目录，发现单文件扩展与包扩展
* :class:`ExtensionDiscovery` — 依赖拓扑排序（Kahn 算法），确保被依赖
  的扩展先加载；缺失依赖与循环依赖记录 WARNING 并跳过

:data:`FRAMEWORK_MODULES` 为框架自身模块名集合，扫描扩展目录时跳过，
避免把框架模块误当作用户扩展。
"""

from __future__ import annotations

import heapq
import logging
from pathlib import Path

from src.extensions.manifest import read_manifest_from_dir

logger = logging.getLogger(__name__)

# 框架自身模块，不应被当作扩展扫描
FRAMEWORK_MODULES = frozenset({
    "base", "loader", "package", "sandbox", "scanner", "security",
    "executable_adapter", "ipc", "manifest",
    "discovery", "security_scan", "ipc_bridge", "resource_monitor",
    "process_manager", "sandbox_proxies", "package_ops", "package_validation",
    "lifecycle", "loader_support",
})


def discover_extensions(directory: Path) -> list[dict]:
    """扫描单个扩展目录，返回发现的扩展信息列表。

    每项形如 ``{"type": "file"|"package", "path": Path}``。
    """
    discovered = []
    for py_file in directory.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        if py_file.stem in FRAMEWORK_MODULES:
            continue
        discovered.append({"type": "file", "path": py_file})

    for subdir in directory.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("_") and (subdir / "__init__.py").exists():
            if subdir.name in FRAMEWORK_MODULES:
                continue
            discovered.append({"type": "package", "path": subdir})

    return discovered


def _extract_extension_deps(
    index: int, ext_info: dict
) -> tuple[str, list[str]]:
    """从扩展信息字典中提取名称与依赖列表。

    对于 package 类型扩展，读取其 ``manifest.yml`` 获取 ``name`` 与
    ``dependencies`` 字段；manifest 缺失或读取失败时回退到目录名作为
    名称、依赖为空。对于 file 类型扩展，使用文件名 stem 作为名称、
    无依赖。

    参数:
        index: 扩展在原列表中的索引（用于生成回退名称）
        ext_info: ``{"type": "file"|"package", "path": Path}``

    返回:
        ``(name, dependencies)`` 元组
    """
    path = ext_info.get("path")
    if ext_info.get("type") == "package" and path is not None:
        manifest = read_manifest_from_dir(path)
        if manifest is not None:
            deps = list(manifest.dependencies or [])
            return manifest.name, deps
        # manifest 缺失：回退到目录名作为名称，无依赖
        return path.name, []
    # file 类型或 path 为空：使用文件名 stem 作为名称，无依赖
    if path is not None:
        return path.stem, []
    return f"ext_{index}", []


class ExtensionDiscovery:
    """扩展加载顺序解析（依赖拓扑排序）。"""

    def resolve_load_order(self, extensions: list[dict]) -> list[dict]:
        """解析扩展加载顺序（依赖排序）。

        基于 manifest 的 ``dependencies`` 字段对扩展进行拓扑排序，
        确保被依赖的扩展先加载。处理两类异常情况：

        * **缺失依赖**：扩展依赖的名称不在已发现扩展集合中时，记录 WARNING
          并跳过该扩展（不包含在返回列表中）。
        * **循环依赖**：检测到环（含自依赖）时，记录 WARNING 并跳过参与环的
          扩展（不包含在返回列表中），不抛出异常。

        无依赖关系的扩展保持原始发现顺序。文件类型扩展（无 manifest）
        视为无依赖，按原顺序参与排序。

        参数:
            extensions: 扩展信息字典列表，
                每项形如 ``{"type": "file"|"package", "path": Path}``

        返回:
            排序后的扩展信息字典列表（被跳过的扩展不包含在内）
        """
        if not extensions:
            return []

        # 1. 收集每个扩展的名称与依赖列表
        ext_infos: list[tuple[str, list[str]]] = []
        for index, ext_info in enumerate(extensions):
            name, deps = _extract_extension_deps(index, ext_info)
            ext_infos.append((name, deps))

        # 2. 构建名称 -> 索引映射（重名时保留第一个）
        name_to_index: dict[str, int] = {}
        for index, (name, _deps) in enumerate(ext_infos):
            if name not in name_to_index:
                name_to_index[name] = index

        # 3. 缺失依赖检测：跳过依赖了不存在扩展的项
        available_names = set(name_to_index.keys())
        skipped_missing: set[int] = set()
        for index, (name, deps) in enumerate(ext_infos):
            missing = [d for d in deps if d not in available_names]
            if missing:
                logger.warning(
                    f"Extension '{name}' has missing dependencies: {missing}; "
                    f"skipping"
                )
                skipped_missing.add(index)

        # 4. 构建邻接表与入度（仅包含未因缺失依赖被跳过的扩展）
        #    边方向：dep -> ext（dep 必须先加载，故从 dep 出边指向 ext）
        active_indices = [
            i for i in range(len(ext_infos)) if i not in skipped_missing
        ]
        in_degree: dict[int, int] = dict.fromkeys(active_indices, 0)
        # adjacency 的值为可变 list，必须用字典推导式为每个键创建独立实例
        adjacency: dict[int, list[int]] = {i: [] for i in active_indices}

        for idx in active_indices:
            _name, deps = ext_infos[idx]
            for dep in deps:
                dep_idx = name_to_index.get(dep)
                # dep_idx 必然存在于 active_indices（缺失依赖的扩展已被跳过）
                if dep_idx is None or dep_idx not in adjacency:
                    continue
                adjacency[dep_idx].append(idx)
                in_degree[idx] += 1

        # 5. Kahn 算法拓扑排序：使用最小堆按原始 index 弹出，保证稳定性
        sorted_indices = self._topo_sort(active_indices, in_degree, adjacency)

        # 6. 循环依赖检测：未被访问的活跃扩展存在循环依赖（含自依赖）
        cyclic_indices = [i for i in active_indices if i not in set(sorted_indices)]
        if cyclic_indices:
            cyclic_names = [ext_infos[i][0] for i in cyclic_indices]
            logger.warning(
                f"Circular dependency detected among extensions: {cyclic_names}; "
                f"skipping these extensions"
            )

        # 7. 按排序后的索引返回原始 extensions 字典
        return [extensions[i] for i in sorted_indices]

    @staticmethod
    def _topo_sort(
        active_indices: list[int],
        in_degree: dict[int, int],
        adjacency: dict[int, list[int]],
    ) -> list[int]:
        """Kahn 拓扑排序：最小堆按原始 index 弹出，保证稳定性。"""
        queue = [i for i in active_indices if in_degree[i] == 0]
        heapq.heapify(queue)
        sorted_indices: list[int] = []
        while queue:
            current = heapq.heappop(queue)
            sorted_indices.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(queue, neighbor)
        return sorted_indices
