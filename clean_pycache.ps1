# PPC8 - [DEPRECATED] Superseded by scripts/clean_caches.py
# That script scans from the repo root (this one scans from the current dir,
# so running it in a subdirectory misses caches elsewhere) and is dry-run by
# default. Kept only for convenience.
Write-Host "[DEPRECATED] Superseded by: python scripts/clean_caches.py" -ForegroundColor Yellow
Write-Host "             (dry-run by default; add --yes to actually delete)" -ForegroundColor Yellow
Write-Host ""

# PPC8 - Clean all __pycache__ directories, .pyc/.pyo files, and tool caches
# Usage: .\clean_pycache.ps1

Write-Host "Cleaning up __pycache__ directories and .pyc/.pyo files..."

Get-ChildItem -Path . -Filter __pycache__ -Recurse -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
        Write-Host "Removing: $($_.FullName)"
        Remove-Item -Recurse -Force $_.FullName
    }

Get-ChildItem -Path . -Include *.pyc,*.pyo -Recurse -File -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Force $_.FullName }

Write-Host ""
Write-Host "Cleaning up ruff cache (.ruff_cache)..."

Get-ChildItem -Path . -Filter .ruff_cache -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    ForEach-Object {
        Write-Host "Removing: $($_.FullName)"
        Remove-Item -Recurse -Force $_.FullName
    }

Write-Host ""
Write-Host "Cleaning up mypy cache (.mypy_cache)..."

Get-ChildItem -Path . -Filter .mypy_cache -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    ForEach-Object {
        Write-Host "Removing: $($_.FullName)"
        Remove-Item -Recurse -Force $_.FullName
    }

Write-Host ""
Write-Host "Cleanup complete."
