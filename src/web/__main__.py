"""生产服务器入口。

用法:
    python -m src.web                      # 默认 0.0.0.0:5000
    python -m src.web --port 8000          # 自定义端口
    python -m src.web --debug              # 开发模式
"""
import argparse

from src.web.app import run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.web",
        description="启动 PPC_NEXT Web 服务（含 WebUI 与 API）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址（默认 127.0.0.1；监听非回环地址需配置 PPC_API_TOKEN）")
    parser.add_argument("--port", type=int, default=5000, help="端口（默认 5000）")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
