"""交互式配置向导 - 分步收集 TTS / 可靠性 / 性能 / 文本 / 章节 / 功能配置。"""

from typing import Any

from rich.prompt import Confirm, Prompt

from src.cli.design import layouts
from src.cli.design.tokens import get_icon

from .branding import BrandAssets


class WizardMixin:
    """配置向导 mixin（config_wizard 交互式向导）。"""

    def config_wizard(self, full: bool = False) -> dict[str, Any] | None:
        """交互式配置向导，使用 StepLayout / ConfigPreviewLayout 输出。"""
        if self.mode != "human":
            self.warning("配置向导需要交互式终端，当前模式不支持")
            return None

        total_steps = 7 if full else 2
        config: dict[str, Any] = {}

        self._wizard_welcome(full)

        self._wizard_step(1, "TTS 核心配置", get_icon("rocket"), total_steps)
        self._wizard_tts_core(config, full)

        if full:
            self._wizard_step(2, "文本分段配置", get_icon("sound"), total_steps)
            self._wizard_segmentation(config)

        self._wizard_step(2 if not full else 3, "可靠性配置", get_icon("running"), total_steps)
        self._wizard_reliability(config, full)

        if full:
            self._wizard_step(4, "性能配置", get_icon("gear"), total_steps)
            self._wizard_performance(config)
            self._wizard_step(5, "文本正则化配置", get_icon("info"), total_steps)
            self._wizard_text_normalization(config)
            self._wizard_step(6, "章节分割配置", get_icon("folder"), total_steps)
            self._wizard_split(config)
            self._wizard_step(7, "输出与网络", get_icon("gear"), total_steps)
            self._wizard_output_network(config)

        return self._wizard_preview_confirm(config)

    def _wizard_step(self, step_num: int, step_title: str, step_icon: str, total_steps: int) -> None:
        layout = layouts.StepLayout(
            step=step_num,
            total=total_steps,
            title=step_title,
            icon=step_icon,
        )
        self.render_layout(layout)

    def _wizard_welcome(self, full: bool) -> None:
        welcome_lines = [
            f"{get_icon('gear')} PPC_NEXT 配置向导",
            f"PPC_NEXT v{BrandAssets.VERSION} | BarbaterLI文本转语音工具",
        ]
        if full:
            welcome_lines.append("完整配置模式：请按照提示完成所有配置。")
        else:
            welcome_lines.append("快速配置模式：仅核心设置。使用 --full 可配置所有项。")

        self.print_panel(
            "\n".join(welcome_lines),
            title="配置向导",
            style="primary",
        )

    def _wizard_tts_core(self, config: dict[str, Any], full: bool) -> None:
        self._wizard_ask_voice(config)
        self._wizard_ask_concurrency(config)
        self._wizard_ask_timeout(config, full)

    def _wizard_ask_voice(self, config: dict[str, Any]) -> None:
        voice_options = [
            ("zh-CN-XiaoxiaoNeural", "晓晓 - 女声，自然流畅"),
            ("zh-CN-YunxiNeural", "云希 - 男声，年轻活力"),
            ("zh-CN-YunjianNeural", "云健 - 男声，成熟稳重"),
            ("zh-CN-XiaoyiNeural", "晓伊 - 女声，温柔甜美"),
            ("zh-CN-YunyangNeural", "云扬 - 男声，新闻播报"),
            ("zh-CN-XiaochenNeural", "晓辰 - 女声，亲切温暖"),
            ("zh-CN-liaoning-Normal", "辽宁 - 地方口音"),
            ("zh-CN-shaanxi-Normal", "陕西 - 地方口音"),
            ("zh-TW-HsiaoChenNeural", "晓臻 - 台湾女声"),
            ("zh-HK-HiuMaanNeural", "晓曼 - 粤语女声"),
        ]

        self.info("可用语音选项:")
        for i, (voice_id, voice_desc) in enumerate(voice_options, 1):
            self.info(f"  {i}. {voice_desc} ({voice_id})")

        voice_choice = Prompt.ask(
            "\n请选择语音",
            choices=[str(i) for i in range(1, len(voice_options) + 1)],
            default="2",
            console=self.console,
        )
        config["tts.voice"] = voice_options[int(voice_choice) - 1][0]

    def _wizard_ask_concurrency(self, config: dict[str, Any]) -> None:
        concurrency = Prompt.ask(
            "并发数 (1-64)",
            default="8",
            console=self.console,
        )
        try:
            config["tts.concurrency"] = max(1, min(64, int(concurrency)))
        except ValueError:
            config["tts.concurrency"] = 8

    def _wizard_output_network(self, config: dict[str, Any]) -> None:
        """输出格式与 TTS 网络/限流配置（full 模式）。写合法 schema 键。"""
        fmt = Prompt.ask(
            "默认音频格式",
            choices=["mp3", "wav", "ogg", "aac"],
            default="mp3",
            console=self.console,
        )
        config["output.default_format"] = fmt

        backend = Prompt.ask(
            "TTS 合成后端",
            choices=["auto", "builtin", "env"],
            default="auto",
            console=self.console,
        )
        config["tts.backend"] = backend

        rate_limit = Prompt.ask(
            "每秒请求限制 (rate_limit, 0=不限制)",
            default="100",
            console=self.console,
        )
        try:
            config["tts.rate_limit"] = max(0, int(rate_limit))
        except ValueError:
            config["tts.rate_limit"] = 100

        proxy = Prompt.ask(
            "TTS 代理地址 (直接回车跳过)",
            default="",
            console=self.console,
        ).strip()
        if proxy:
            config["tts.proxy"] = proxy

    def _wizard_ask_timeout(self, config: dict[str, Any], full: bool) -> None:
        timeout_mode_options = [
            ("auto", "自动 - 根据文本长度动态调整"),
            ("fixed", "固定 - 使用固定超时值"),
            ("adaptive", "自适应 - 基于历史记录调整"),
        ]
        self.info("超时模式选项:")
        for i, (mode_id, mode_desc) in enumerate(timeout_mode_options, 1):
            self.info(f"  {i}. {mode_desc} ({mode_id})")

        timeout_choice = Prompt.ask(
            "\n请选择超时模式",
            choices=[str(i) for i in range(1, len(timeout_mode_options) + 1)],
            default="1",
            console=self.console,
        )
        config["tts.timeout_mode"] = timeout_mode_options[int(timeout_choice) - 1][0]

        timeout = Prompt.ask(
            "固定超时时间 (秒, 0=自动推导)",
            default="0",
            console=self.console,
        )
        try:
            config["tts.timeout"] = max(0, int(timeout))
        except ValueError:
            config["tts.timeout"] = 0

        if full:
            timeout_min = Prompt.ask(
                "最小超时时间 (秒)",
                default="45",
                console=self.console,
            )
            try:
                config["tts.timeout_min"] = max(10, min(450, int(timeout_min)))
            except ValueError:
                config["tts.timeout_min"] = 45

            timeout_max = Prompt.ask(
                "最大超时时间 (秒)",
                default="900",
                console=self.console,
            )
            try:
                config["tts.timeout_max"] = max(60, min(3600, int(timeout_max)))
            except ValueError:
                config["tts.timeout_max"] = 900

    def _wizard_segmentation(self, config: dict[str, Any]) -> None:
        enable_segmentation = Confirm.ask(
            "\n启用文本分段?",
            default=True,
            console=self.console,
        )
        config["tts.enable_segmentation"] = enable_segmentation

        max_seg_len = Prompt.ask(
            "最大分段长度 (字符数)",
            default="2500",
            console=self.console,
        )
        try:
            config["tts.max_segment_length"] = max(100, int(max_seg_len))
        except ValueError:
            config["tts.max_segment_length"] = 2500

        min_seg_len = Prompt.ask(
            "最小分段长度 (字符数)",
            default="100",
            console=self.console,
        )
        try:
            config["tts.min_segment_length"] = max(10, min(1000, int(min_seg_len)))
        except ValueError:
            config["tts.min_segment_length"] = 100

        silence_ms = Prompt.ask(
            "分段间静音时长 (毫秒)",
            default="100",
            console=self.console,
        )
        try:
            config["tts.segment_silence_ms"] = max(0, min(1000, int(silence_ms)))
        except ValueError:
            config["tts.segment_silence_ms"] = 100

    def _wizard_reliability(self, config: dict[str, Any], full: bool) -> None:
        max_retries = Prompt.ask(
            "TTS 最大重试次数 (0-20)",
            default="3",
            console=self.console,
        )
        try:
            retries_value = max(0, min(20, int(max_retries)))
            config["reliability.tts_retry.max_retries"] = retries_value
            config["tts.retries"] = retries_value
        except ValueError:
            config["reliability.tts_retry.max_retries"] = 3
            config["tts.retries"] = 3

        base_delay = Prompt.ask(
            "重试基础延迟 (秒)",
            default="1.0",
            console=self.console,
        )
        try:
            config["reliability.tts_retry.base_delay"] = max(0.1, min(60.0, float(base_delay)))
        except ValueError:
            config["reliability.tts_retry.base_delay"] = 1.0

        if full:
            max_delay = Prompt.ask(
                "重试最大延迟 (秒)",
                default="60.0",
                console=self.console,
            )
            try:
                config["reliability.tts_retry.max_delay"] = max(1.0, min(300.0, float(max_delay)))
            except ValueError:
                config["reliability.tts_retry.max_delay"] = 60.0

            circuit_threshold = Prompt.ask(
                "熔断器失败阈值",
                default="5",
                console=self.console,
            )
            try:
                config["reliability.tts_circuit.failure_threshold"] = max(1, min(20, int(circuit_threshold)))
            except ValueError:
                config["reliability.tts_circuit.failure_threshold"] = 5

    def _wizard_performance(self, config: dict[str, Any]) -> None:
        rate_limit = Prompt.ask(
            "每秒请求数限制",
            default="100",
            console=self.console,
        )
        try:
            config["tts.rate_limit"] = max(1, int(rate_limit))
        except ValueError:
            config["tts.rate_limit"] = 100

        ema_alpha = Prompt.ask(
            "EMA 平滑因子 (0.0-1.0)",
            default="0.3",
            console=self.console,
        )
        try:
            config["tts.ema_alpha"] = max(0.0, min(1.0, float(ema_alpha)))
        except ValueError:
            config["tts.ema_alpha"] = 0.3

        buffer_size = Prompt.ask(
            "缓冲区大小",
            default="32",
            console=self.console,
        )
        try:
            config["tts.buffer_size"] = max(1, int(buffer_size))
        except ValueError:
            config["tts.buffer_size"] = 32

    def _wizard_text_normalization(self, config: dict[str, Any]) -> None:
        enable_text_norm = Confirm.ask(
            "\n启用文本正则化?",
            default=True,
            console=self.console,
        )
        config["tts.text_normalization.enable_text_normalization"] = enable_text_norm

        if enable_text_norm:
            enable_whitespace = Confirm.ask(
                "启用空白字符规范化?",
                default=True,
                console=self.console,
            )
            config["tts.text_normalization.enable_whitespace_normalization"] = enable_whitespace

            enable_linebreak = Confirm.ask(
                "启用换行符规范化?",
                default=True,
                console=self.console,
            )
            config["tts.text_normalization.enable_linebreak_normalization"] = enable_linebreak

            enable_punct = Confirm.ask(
                "启用标点符号规范化?",
                default=True,
                console=self.console,
            )
            config["tts.text_normalization.enable_punctuation_normalization"] = enable_punct

            enable_trim = Confirm.ask(
                "启用行首尾空白去除?",
                default=True,
                console=self.console,
            )
            config["tts.text_normalization.enable_trim_whitespace"] = enable_trim

    def _wizard_split(self, config: dict[str, Any]) -> None:
        split_presets = [
            ("chinese_novel", "中文小说"),
            ("english_novel", "英文小说"),
            ("default", "默认"),
        ]
        self.info("章节预设选项:")
        for i, (preset_id, preset_desc) in enumerate(split_presets, 1):
            self.info(f"  {i}. {preset_desc} ({preset_id})")

        split_choice = Prompt.ask(
            "\n请选择章节预设",
            choices=[str(i) for i in range(1, len(split_presets) + 1)],
            default="1",
            console=self.console,
        )
        config["split.preset"] = split_presets[int(split_choice) - 1][0]

        min_chap_len = Prompt.ask(
            "最小章节长度 (字符数)",
            default="100",
            console=self.console,
        )
        try:
            config["split.min_chapter_length"] = max(10, int(min_chap_len))
        except ValueError:
            config["split.min_chapter_length"] = 100

        add_sep = Confirm.ask(
            "在章节名后添加等于号分隔符?",
            default=True,
            console=self.console,
        )
        config["split.add_title_separator"] = add_sep

    def _wizard_preview_confirm(self, config: dict[str, Any]) -> dict[str, Any] | None:
        self.print_panel("配置预览", title="配置预览", style="primary")

        config_labels = {
            "tts.voice": "语音模型",
            "tts.concurrency": "并发数",
            "tts.retries": "TTS重试次数",
            "tts.timeout_mode": "超时模式",
            "tts.timeout": "固定超时(秒)",
            "tts.timeout_min": "最小超时(秒)",
            "tts.timeout_max": "最大超时(秒)",
            "tts.enable_segmentation": "启用分段",
            "tts.max_segment_length": "最大分段长度",
            "tts.min_segment_length": "最小分段长度",
            "tts.segment_silence_ms": "分段静音(ms)",
            "reliability.tts_retry.max_retries": "可靠性重试次数",
            "reliability.tts_retry.base_delay": "重试基础延迟(秒)",
            "reliability.tts_retry.max_delay": "重试最大延迟(秒)",
            "reliability.tts_circuit.failure_threshold": "熔断器失败阈值",
            "tts.rate_limit": "每秒请求限制",
            "tts.ema_alpha": "EMA平滑因子",
            "tts.buffer_size": "缓冲区大小",
            "tts.text_normalization.enable_text_normalization": "启用文本正则化",
            "tts.text_normalization.enable_whitespace_normalization": "空白规范化",
            "tts.text_normalization.enable_linebreak_normalization": "换行规范化",
            "tts.text_normalization.enable_punctuation_normalization": "标点规范化",
            "tts.text_normalization.enable_trim_whitespace": "空白去除",
            "split.preset": "章节预设",
            "split.min_chapter_length": "最小章节长度",
            "split.add_title_separator": "章节名分隔符",
        }

        preview_layout = layouts.ConfigPreviewLayout(config=config, labels=config_labels)
        self.render_layout(preview_layout)

        if Confirm.ask(
            "\n确认保存配置?",
            default=True,
            console=self.console,
        ):
            self.success("配置已保存！")
            return config
        else:
            self.warning("配置已取消")
            return None
