@echo off
REM PPC8 - Clean all __pycache__ directories and .pyc files
REM Usage: clean_pycache.bat

echo Cleaning up __pycache__ directories and .pyc files...

for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        echo Removing: %%d
        rmdir /s /q "%%d"
    )
)

del /s /q *.pyc >nul 2>&1
del /s /q *.pyo >nul 2>&1

echo Cleanup complete.
pause
