from src.web.handlers.convert_handler import ConvertTaskHandler
from src.web.handlers.fanqie_handler import FanqieInstallTaskHandler, FanqieUpdateTaskHandler
from src.web.task_queue import TaskManager

__all__ = [
    "ConvertTaskHandler",
    "FanqieInstallTaskHandler",
    "FanqieUpdateTaskHandler",
    "register_default_handlers",
]


def register_default_handlers(manager: TaskManager) -> None:
    manager.register_handler(ConvertTaskHandler(manager))
    manager.register_handler(FanqieInstallTaskHandler(manager))
    manager.register_handler(FanqieUpdateTaskHandler(manager))
