import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    secret_key: str = ""
    cors_origins: str = "*"


def get_web_config() -> WebConfig:
    config = WebConfig()

    config.host = os.environ.get("PPC10_WEB_HOST", config.host)
    config.port = int(os.environ.get("PPC10_WEB_PORT", str(config.port)))
    config.debug = os.environ.get("PPC10_WEB_DEBUG", "").lower() in ("true", "1", "yes")
    config.secret_key = os.environ.get("PPC10_WEB_SECRET_KEY", config.secret_key)
    config.cors_origins = os.environ.get("PPC10_WEB_CORS_ORIGINS", config.cors_origins)

    try:
        from src.config.manager import ConfigManager

        manager = ConfigManager()
        ppc10_config = manager.get_config()

        if hasattr(ppc10_config, "web"):
            web = ppc10_config.web
            if hasattr(web, "host"):
                config.host = web.host or config.host
            if hasattr(web, "port"):
                config.port = web.port or config.port
            if hasattr(web, "debug"):
                config.debug = web.debug
            if hasattr(web, "secret_key"):
                config.secret_key = web.secret_key or config.secret_key
            if hasattr(web, "cors_origins"):
                config.cors_origins = web.cors_origins or config.cors_origins
    except Exception:
        logger.debug("无法从PPC10 配置加载 Web 配置，使用默认值和环境变量", exc_info=True)

    return config
