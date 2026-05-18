#!/usr/bin/env python3
"""
代码导出工具 - 将项目所有代码导出为一个结构化的Markdown文件
包含目录树结构和代码块
"""

import os
import sys
from pathlib import Path
from typing import List, Set

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class CodeExporter:
    """代码导出器"""

    EXCLUDE_DIRS = {
        '__pycache__', '.git', '.venv', 'venv', 'node_modules',
        '.pytest_cache', '.mypy_cache', '.trae'
    }

    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
        '.json', '.yaml', '.yml', '.toml', '.md', '.txt',
        '.sh', '.bat', '.cfg', '.ini', '.xml', '.sql'
    }

    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown',
        '.txt': 'text',
        '.sh': 'bash',
        '.bat': 'batch',
        '.cfg': 'ini',
        '.ini': 'ini',
        '.xml': 'xml',
        '.sql': 'sql',
    }

    def __init__(self, root_path: str, output_path: str):
        self.root = Path(root_path).resolve()
        self.output = Path(output_path)

    def get_extension(self, filename: str) -> str:
        return Path(filename).suffix.lower()

    def should_include_file(self, file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        if file_path.name.startswith('.'):
            return False
        ext = self.get_extension(file_path.name)
        return ext in self.CODE_EXTENSIONS

    def should_include_dir(self, dir_path: Path) -> bool:
        return dir_path.name not in self.EXCLUDE_DIRS and not dir_path.name.startswith('.')

    def get_project_tree(self) -> str:
        tree_parts = []
        tree_parts.append("```")
        tree_parts.append(self._build_tree(self.root, prefix="", is_last=True))
        tree_parts.append("```")
        return "\n".join(tree_parts)

    def _build_tree(self, directory: Path, prefix: str = "", is_last: bool = True) -> str:
        lines = []

        if prefix == "":
            lines.append(f"{directory.name}/")

        items = sorted([
            d for d in directory.iterdir()
            if d.is_dir() and self.should_include_dir(d)
        ], key=lambda x: x.name)

        files = sorted([
            f for f in directory.iterdir()
            if f.is_file() and self.should_include_file(f)
        ], key=lambda x: x.name)

        all_items = items + files
        count = len(all_items)

        for idx, item in enumerate(all_items):
            is_last_item = (idx == count - 1)
            connector = "└── " if is_last_item else "├── "

            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                extension = "    " if is_last_item else "│   "
                lines.append(self._build_tree(item, prefix + extension, is_last_item))
            else:
                lines.append(f"{prefix}{connector}{item.name}")

        return "\n".join(lines)

    def get_file_content(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception:
                return "[无法读取文件内容]"
        except Exception as e:
            return f"[读取文件时出错: {e}]"

    def get_language_tag(self, file_path: Path) -> str:
        ext = self.get_extension(file_path.name)
        return self.LANGUAGE_MAP.get(ext, '')

    def collect_files(self) -> List[Path]:
        files = []
        for root, dirs, filenames in os.walk(self.root):
            dirs[:] = [d for d in dirs if self.should_include_dir(Path(root) / d)]

            for filename in sorted(filenames):
                file_path = Path(root) / filename
                if self.should_include_file(file_path):
                    files.append(file_path)

        return sorted(files, key=lambda x: str(x.relative_to(self.root)))

    def generate_markdown(self) -> str:
        md_content = []

        md_content.append("# 项目代码总览\n")
        md_content.append(f"**项目路径**: `{self.root}`\n")
        md_content.append("---\n")

        md_content.append("## 📁 项目目录结构\n")
        md_content.append(self.get_project_tree())
        md_content.append("\n---\n")

        md_content.append("## 💻 源代码文件\n")

        files = self.collect_files()
        total_files = len(files)

        md_content.append(f"共找到 **{total_files}** 个代码文件\n")
        md_content.append("---\n")

        current_dir = None

        for idx, file_path in enumerate(files, 1):
            relative_path = file_path.relative_to(self.root)
            relative_str = str(relative_path).replace('\\', '/')

            file_dir = str(relative_path.parent)
            if file_dir != current_dir:
                current_dir = file_dir
                md_content.append(f"\n### 📂 {current_dir}\n")

            md_content.append(f"\n#### {idx}. [{relative_str}](file:///{file_path})\n")

            language = self.get_language_tag(file_path)
            content = self.get_file_content(file_path)

            md_content.append(f"```{language}")
            md_content.append(content)
            if not content.endswith('\n'):
                md_content.append("")
            md_content.append("```\n")
            md_content.append("---\n")

        md_content.append(f"\n**导出完成** - 共导出 {total_files} 个文件\n")

        return "\n".join(md_content)

    def export(self):
        print(f"🔍 扫描项目: {self.root}")
        files = self.collect_files()
        print(f"📊 找到 {len(files)} 个代码文件")

        print("📝 生成Markdown文件...")
        md_content = self.generate_markdown()

        print(f"💾 保存到: {self.output}")
        with open(self.output, 'w', encoding='utf-8') as f:
            f.write(md_content)

        file_size = self.output.stat().st_size
        print(f"✅ 导出完成! 文件大小: {file_size / 1024:.1f} KB")


def main():
    root_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(root_path, "代码总体.md")

    exporter = CodeExporter(root_path, output_path)
    exporter.export()


if __name__ == "__main__":
    main()
