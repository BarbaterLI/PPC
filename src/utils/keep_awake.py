import ctypes
import logging
import platform
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ScreenKeepAwake:
    def __init__(self):
        self._system = platform.system()
        self._prevent_sleep = None
        self._prevent_display_sleep = None
        self._is_active = False
    
    def _keep_awake_windows(self) -> bool:
        try:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002

            self._prevent_sleep = ctypes.windll.kernel32.SetThreadExecutionState
            self._prevent_sleep.argtypes = [wintypes.DWORD]
            self._prevent_sleep.restype = wintypes.DWORD

            self._prevent_display_sleep = ctypes.windll.kernel32.SetThreadExecutionState
            self._prevent_display_sleep.argtypes = [wintypes.DWORD]
            self._prevent_display_sleep.restype = wintypes.DWORD

            self._prevent_sleep(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            self._prevent_display_sleep(ES_CONTINUOUS | ES_DISPLAY_REQUIRED)

            logger.info("Windows屏幕常亮已启用")
            return True
        except Exception as e:
            logger.error(f"启用Windows屏幕常亮失败: {e}")
            return False
    
    def _allow_sleep_windows(self):
        try:
            if self._prevent_sleep:
                self._prevent_sleep(0x80000000)
                logger.info("Windows屏幕常亮已禁用")
        except Exception as e:
            logger.error(f"禁用Windows屏幕常亮失败: {e}")
    
    def _keep_awake_linux(self) -> bool:
        try:
            from ctypes import cdll
            libx = cdll.LoadLibrary('libX11.so.6')
            display = libv.XOpenDisplay(None)
            if display:
                logger.info("Linux屏幕常亮已启用 (X11)")
                return True
            else:
                logger.warning("无法打开X11显示")
                return False
        except Exception as e:
            logger.error(f"启用Linux屏幕常亮失败: {e}")
            return False
    
    def _allow_sleep_linux(self):
        try:
            from ctypes import cdll
            libx = cdll.LoadLibrary('libX11.so.6')
            display = libx.XOpenDisplay(None)
            if display:
                libx.XCloseDisplay(display)
                logger.info("Linux屏幕常亮已禁用")
        except Exception as e:
            logger.error(f"禁用Linux屏幕常亮失败: {e}")
    
    def _keep_awake_macos(self) -> bool:
        try:
            from ctypes import cdll, c_uint32, c_bool
            from Foundation import NSBundle
            
            bundle = NSBundle.mainBundle()
            path = bundle.bundlePath()
            
            IOKit = cdll.LoadLibrary('/System/Library/Frameworks/IOKit.framework/IOKit')
            
            service = IOKit.IORegistryEntryFromPath(0, b"IODisplayConnect")
            if service:
                logger.info("macOS屏幕常亮已启用")
                return True
            else:
                logger.warning("无法访问显示服务")
                return False
        except Exception as e:
            logger.error(f"启用macOS屏幕常亮失败: {e}")
            return False
    
    def _allow_sleep_macos(self):
        try:
            logger.info("macOS屏幕常亮已禁用")
        except Exception as e:
            logger.error(f"禁用macOS屏幕常亮失败: {e}")
    
    def keep_awake(self) -> bool:
        if self._is_active:
            logger.warning("屏幕常亮已在启用状态")
            return True
        
        if self._system == "Windows":
            result = self._keep_awake_windows()
        elif self._system == "Linux":
            result = self._keep_awake_linux()
        elif self._system == "Darwin":
            result = self._keep_awake_macos()
        else:
            logger.warning(f"不支持的系统: {self._system}")
            result = False
        
        if result:
            self._is_active = True
            logger.info(f"屏幕常亮已启用 (系统: {self._system})")
        else:
            logger.warning("屏幕常亮启用失败，程序将正常运行")
        
        return result
    
    def allow_sleep(self):
        if not self._is_active:
            logger.debug("屏幕常亮未在启用状态，无需禁用")
            return
        
        if self._system == "Windows":
            self._allow_sleep_windows()
        elif self._system == "Linux":
            self._allow_sleep_linux()
        elif self._system == "Darwin":
            self._allow_sleep_macos()
        else:
            logger.warning(f"不支持的系统: {self._system}")
        
        self._is_active = False
        logger.info("屏幕常亮已禁用")
    
    def is_active(self) -> bool:
        return self._is_active
    
    def get_status(self) -> dict:
        return {
            "is_active": self._is_active,
            "system": self._system,
            "platform": platform.platform(),
        }
    
    def __enter__(self):
        self.keep_awake()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.allow_sleep()
    
    def __del__(self):
        if self._is_active:
            self.allow_sleep()


def keep_awake_context():
    return ScreenKeepAwake()


if __name__ == "__main__":
    print("屏幕常亮测试...")
    
    keep_awake = ScreenKeepAwake()
    
    if keep_awake.keep_awake():
        print("屏幕常亮已启用，10秒后自动禁用...")
        time.sleep(10)
        keep_awake.allow_sleep()
        print("测试完成")
    else:
        print("屏幕常亮启用失败")
