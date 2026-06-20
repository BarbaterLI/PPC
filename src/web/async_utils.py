import asyncio
import functools


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def async_route(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return run_async(func(*args, **kwargs))

    return wrapper
