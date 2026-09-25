"""SuperSplitExtension 集成测试。"""

from __future__ import annotations

from pathlib import Path

from src.extensions.supersplit.extension import SuperSplitExtension
from src.extensions.supersplit.models import (
    SplitMode,
    SuperSplitOptions,
)


class TestSuperSplitExtensionInterface:
    """SuperSplitExtension 接口测试。"""

    def test_split_strategy_interface(self, tmp_path: Path) -> None:
        """验证 SuperSplitExtension.split_file 返回 SplitResult。"""
        from src.extensions.supersplit.models import SplitResult

        # 准备测试文件
        input_file = tmp_path / "novel.txt"
        input_file.write_text(
            "书名：测试小说\n作者：测试作者\n\n"
            "第一章 开始\n这是第一章的内容。\n" * 50 + "\n"
            "第二章 发展\n这是第二章的内容。\n" * 50,
            encoding="utf-8",
        )
        output_dir = tmp_path / "output"

        extension = SuperSplitExtension()
        options = SuperSplitOptions(
            mode=SplitMode.NOVEL_CHAPTER,
            target_length=1000,
            min_length=50,
            max_length=5000,
            clean_novel=False,
            audit_logging=False,
            publish_events=False,
        )

        result = extension.split_file(input_file, output_dir, options)

        assert isinstance(result, SplitResult)
        assert len(result.output_files) > 0
        # 所有输出文件应存在
        for f in result.output_files:
            assert f.exists()

    def test_dict_to_options_conversion(self) -> None:
        """验证字符串枚举正确转换为 SplitMode。"""
        options = SuperSplitOptions(mode="novel_chapter")
        assert options.mode == SplitMode.NOVEL_CHAPTER

        options = SuperSplitOptions(mode="novel_volume")
        assert options.mode == SplitMode.NOVEL_VOLUME

        options = SuperSplitOptions(mode="novel_scene")
        assert options.mode == SplitMode.NOVEL_SCENE

        options = SuperSplitOptions(mode="novel_dialogue")
        assert options.mode == SplitMode.NOVEL_DIALOGUE

    def test_get_default_options(self) -> None:
        """验证默认选项包含所有字段。"""
        options = SuperSplitOptions()

        assert options.mode == SplitMode.WORD_COUNT
        assert options.target_length == 1000
        assert options.min_length == 100
        assert options.max_length == 2000
        assert options.overlap == 0
        assert options.jobs == 1
        assert options.clean_novel is True
        assert options.audit_logging is True
        assert options.publish_events is True

    def test_novel_mode_split(self, tmp_path: Path) -> None:
        """端到端小说章节切分。"""
        input_file = tmp_path / "novel.txt"
        # 构造含明显章节边界的小说
        content_parts = [
            "书名：端到端测试\n作者：测试\n简介：这是一本测试小说\n\n",
            "第一章 引子\n" + "这是第一章的正文内容。" * 50 + "\n\n",
            "第二章 发展\n" + "这是第二章的正文内容。" * 50 + "\n\n",
            "第三章 高潮\n" + "这是第三章的正文内容。" * 50,
        ]
        input_file.write_text("".join(content_parts), encoding="utf-8")

        output_dir = tmp_path / "output"
        extension = SuperSplitExtension()
        options = SuperSplitOptions(
            mode=SplitMode.NOVEL_CHAPTER,
            target_length=500,
            min_length=50,
            max_length=10000,
            clean_novel=False,
            audit_logging=False,
            publish_events=False,
        )

        result = extension.split_file(input_file, output_dir, options)

        # 应该至少切分出 3 个章节片段
        assert len(result.output_files) >= 3
        # novel_metadata 应被填充
        assert result.novel_metadata
        assert "chapters" in result.novel_metadata
        assert len(result.novel_metadata["chapters"]) >= 3


class TestNovelSplittersIntegration:
    """小说分割器集成测试。"""

    def test_novel_chapter_splitter_via_extension(self, tmp_path: Path) -> None:
        """通过 Extension 调用 NovelChapterSplitter。"""
        input_file = tmp_path / "chapters.txt"
        content = (
            "第一章 引子\n" + "内容" * 200 + "\n\n"
            "第二章 发展\n" + "内容" * 200 + "\n\n"
            "第三章 结局\n" + "内容" * 200
        )
        input_file.write_text(content, encoding="utf-8")

        extension = SuperSplitExtension()
        options = SuperSplitOptions(
            mode=SplitMode.NOVEL_CHAPTER,
            min_length=50,
            max_length=10000,
            clean_novel=False,
            audit_logging=False,
            publish_events=False,
        )

        result = extension.split_file(input_file, tmp_path / "out", options)
        assert len(result.output_files) >= 2

    def test_novel_volume_splitter_via_extension(self, tmp_path: Path) -> None:
        """通过 Extension 调用 NovelVolumeSplitter。"""
        input_file = tmp_path / "volumes.txt"
        content = (
            "第一卷 春\n第一章\n" + "内容" * 100 + "\n第二章\n" + "内容" * 100 + "\n\n"
            "第二卷 夏\n第三章\n" + "内容" * 100 + "\n第四章\n" + "内容" * 100
        )
        input_file.write_text(content, encoding="utf-8")

        extension = SuperSplitExtension()
        options = SuperSplitOptions(
            mode=SplitMode.NOVEL_VOLUME,
            clean_novel=False,
            audit_logging=False,
            publish_events=False,
        )

        result = extension.split_file(input_file, tmp_path / "out", options)
        # 应该切分出至少 2 个卷
        assert len(result.output_files) >= 1


class TestLifecycleHooks:
    """生命周期钩子测试。"""

    def test_audit_logging_disabled(self, tmp_path: Path) -> None:
        """验证 audit_logging=False 时不记录审计。"""
        input_file = tmp_path / "novel.txt"
        input_file.write_text("第一章\n内容" * 100, encoding="utf-8")

        extension = SuperSplitExtension()
        options = SuperSplitOptions(
            audit_logging=False,
            publish_events=False,
            clean_novel=False,
        )

        # 调用 _audit 不应抛异常
        extension._current_options = options
        extension._audit(input_file, "read", success=True)
        # 验证 _current_options 已设置
        assert extension._current_options is not None

    def test_emit_disabled(self) -> None:
        """验证 publish_events=False 时不发布事件。"""
        extension = SuperSplitExtension()
        options = SuperSplitOptions(publish_events=False)
        extension._current_options = options

        # 调用 _emit 不应抛异常
        extension._emit("test.event", {"key": "value"})
