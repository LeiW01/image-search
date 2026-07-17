from fashion_search.app import build_app
from fashion_search.cli import build_parser


def test_cli_help_contains_all_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("inspect", "index", "calibrate", "evaluate", "serve"):
        assert command in help_text


def test_gradio_app_builds_without_loading_models() -> None:
    app = build_app(None, status_message="请先运行建索引命令")
    assert app is not None
