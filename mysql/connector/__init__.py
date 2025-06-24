import sys
import os

from mysql.connector.cursor import _CursorProxy

_original_driver_module = None


def _get_underlying_driver():
    global _original_driver_module
    if _original_driver_module:
        return _original_driver_module

    current_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    path_was_present = current_package_dir in sys.path
    if path_was_present:
        sys.path.remove(current_package_dir)

    try:
        import mysql.connector as driver
        _original_driver_module = driver
    finally:
        if path_was_present:
            sys.path.insert(0, current_package_dir)

    return _original_driver_module


class _ConnectionProxy:
    def __init__(self, real_connection, config):
        self._real_connection = real_connection
        self._config = config

    def cursor(self, *args, **kwargs):
        real_cursor = self._real_connection.cursor(*args, **kwargs)
        return _CursorProxy(real_cursor, self._config)

    def __getattr__(self, name):
        return getattr(self._real_connection, name)


def connect(*args, **kwargs):
    driver = _get_underlying_driver()
    real_connection = driver.connect(*args, **kwargs)
    return _ConnectionProxy(real_connection, kwargs)


_driver = _get_underlying_driver()
for attr_name in dir(_driver):
    if not hasattr(sys.modules[__name__], attr_name):
        setattr(sys.modules[__name__], attr_name, getattr(_driver, attr_name))
