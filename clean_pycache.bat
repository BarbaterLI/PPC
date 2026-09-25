@echo off
REM PPC8 - Clean all __pycache__ directories, .pyc/.pyo files, and tool caches
REM Usage: clean_pycache.bat

echo Cleaning up __pycache__ directories and .pyc/.pyo files...

for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        echo Removing: %%d
        rmdir /s /q "%%d"
    )
)

del /s /q *.pyc >nul 2>&1
del /s /q *.pyo >nul 2>&1

echo.
echo Cleaning up ruff cache (.ruff_cache)...

for /d /r %%d in (.ruff_cache) do (
    if exist "%%d" (
        echo Removing: %%d
        rmdir /s /q "%%d"
    )
)

echo.
echo Cleaning up mypy cache (.mypy_cache)...

for /d /r %%d in (.mypy_cache) do (
    if exist "%%d" (
        echo Removing: %%d
        rmdir /s /q "%%d"
    )
)

echo.
echo Cleanup complete.
pause
