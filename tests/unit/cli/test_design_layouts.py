"""Tests for src.cli.design.layouts."""

from __future__ import annotations

from src.cli.design import atoms, layouts


class TestWelcomeLayout:
    def test_to_components_returns_expected_types(self):
        layout = layouts.WelcomeLayout(
            version="1.0.0",
            tagline="Test tagline",
            commands=[("cmd1", "desc1"), ("cmd2", "desc2")],
            tips=["tip1", "tip2"],
        )
        components = layout.to_components()

        assert len(components) == 3
        assert isinstance(components[0], atoms.Panel)
        assert isinstance(components[1], atoms.Table)
        assert isinstance(components[2], atoms.Panel)

    def test_to_components_content(self):
        layout = layouts.WelcomeLayout(
            version="1.0.0",
            tagline="Test tagline",
            commands=[("cmd1", "desc1")],
            tips=["tip1"],
        )
        panel, table, tips_panel = layout.to_components()

        assert "1.0.0" in panel._content_text()
        assert "Test tagline" in panel._content_text()
        assert table.rows == [["cmd1", "desc1"]]
        assert "tip1" in tips_panel._content_text()


class TestCompletionReportLayout:
    def test_success_rate_computed_correctly(self):
        layout = layouts.CompletionReportLayout(
            total=10,
            completed=7,
            failed=3,
            elapsed=12.5,
        )
        assert layout.success_rate == 70.0

    def test_success_rate_zero_when_total_zero(self):
        layout = layouts.CompletionReportLayout(
            total=0,
            completed=0,
            failed=0,
            elapsed=0.0,
        )
        assert layout.success_rate == 0.0

    def test_includes_error_table_when_errors_exist(self):
        layout = layouts.CompletionReportLayout(
            total=10,
            completed=7,
            failed=3,
            elapsed=12.5,
            error_type_counts={"TimeoutError": 2, "ValueError": 1},
        )
        components = layout.to_components()

        assert len(components) == 2
        assert isinstance(components[0], atoms.Panel)
        assert isinstance(components[1], atoms.Table)
        assert components[1].headers == ["错误类型", "数量", "占比"]
        assert ["TimeoutError", 2, "66.7%"] in components[1].rows

    def test_no_error_table_when_empty(self):
        layout = layouts.CompletionReportLayout(
            total=10,
            completed=10,
            failed=0,
            elapsed=12.5,
        )
        components = layout.to_components()
        assert len(components) == 1
        assert isinstance(components[0], atoms.Panel)


class TestErrorLayout:
    def test_includes_code_and_message(self):
        layout = layouts.ErrorLayout(
            code="E_TEST",
            message="Something went wrong",
        )
        components = layout.to_components()

        assert len(components) == 1
        panel = components[0]
        assert isinstance(panel, atoms.Panel)
        content = panel._content_text()
        assert "E_TEST" in content
        assert "Something went wrong" in content

    def test_includes_hint_and_suggestions(self):
        layout = layouts.ErrorLayout(
            code="E_HINT",
            message="Error with hint",
            hint="Try this",
            suggestions=["suggestion A", "suggestion B"],
        )
        components = layout.to_components()

        panel = components[0]
        content = panel._content_text()
        assert "Try this" in content
        assert "suggestion A" in content
        assert "suggestion B" in content

    def test_includes_trace_when_verbose_and_cause(self):
        cause = ValueError("root cause")
        layout = layouts.ErrorLayout(
            code="E_TRACE",
            message="With trace",
            verbose=True,
            cause=cause,
        )
        components = layout.to_components()

        assert len(components) == 2
        assert isinstance(components[0], atoms.Panel)
        assert isinstance(components[1], atoms.Trace)
        assert components[1].exception is cause


class TestConfigPreviewLayout:
    def test_maps_labels_correctly(self):
        layout = layouts.ConfigPreviewLayout(
            config={"key1": "value1", "key2": True},
            labels={"key1": "Friendly Key 1", "key2": "Friendly Key 2"},
        )
        components = layout.to_components()

        assert len(components) == 1
        table = components[0]
        assert isinstance(table, atoms.Table)
        assert table.headers == ["配置项", "值"]
        assert ["Friendly Key 1", "value1"] in table.rows
        assert ["Friendly Key 2", "是"] in table.rows

    def test_uses_key_as_fallback_label(self):
        layout = layouts.ConfigPreviewLayout(
            config={"unknown_key": 42},
        )
        components = layout.to_components()
        table = components[0]

        assert ["unknown_key", "42"] in table.rows


class TestTaskDashboardLayout:
    def test_to_components_includes_progress_and_stats(self):
        layout = layouts.TaskDashboardLayout(
            total=10,
            completed=4,
            failed=1,
            current_task="task-1",
            speed=2.5,
            elapsed=30.0,
            eta=60.0,
        )
        components = layout.to_components()

        assert isinstance(components[0], atoms.Message)
        assert isinstance(components[1], atoms.ProgressBar)
        assert isinstance(components[2], atoms.StatGrid)
        assert components[1].current == 5
        assert components[1].total == 10
        assert "task-1" in components[2].items["当前任务"]


class TestCommandHelpLayout:
    def test_to_components_returns_command_help_atom(self):
        layout = layouts.CommandHelpLayout(
            command="convert",
            description="转换文本为语音",
            usage="ppc10 convert <input> [output]",
            examples=[{"command": "ppc10 convert ./txt ./out", "description": "基本用法"}],
            options=[{"name": "--voice", "description": "语音模型"}],
            see_also=["batch"],
        )
        components = layout.to_components()

        assert len(components) == 1
        assert isinstance(components[0], atoms.CommandHelp)
        assert components[0].command == "convert"

    def test_to_components_with_empty_optional_fields(self):
        layout = layouts.CommandHelpLayout(
            command="voices",
            description="列出语音",
            usage="ppc10 voices",
        )
        components = layout.to_components()

        assert len(components) == 1
        assert isinstance(components[0], atoms.CommandHelp)
        assert components[0].examples == []
        assert components[0].options == []
        assert components[0].see_also == []


class TestStepLayout:
    def test_to_components_returns_message_and_progress(self):
        layout = layouts.StepLayout(
            step=2,
            total=5,
            title="Test Step",
            icon="⚙",
        )
        components = layout.to_components()

        assert len(components) == 2
        assert isinstance(components[0], atoms.Message)
        assert isinstance(components[1], atoms.ProgressBar)
        assert components[1].current == 2
        assert components[1].total == 5
        assert "Test Step" in components[0].text
