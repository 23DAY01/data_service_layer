from .cursor import _CursorProxy


class _ConnectionProxy:
    def __init__(self, real_connection, config):
        self._real_connection = real_connection
        self._config = config

    def cursor(self, *args, **kwargs):
        real_cursor = self._real_connection.cursor(*args, **kwargs)
        return _CursorProxy(real_cursor, self._config)

    def __getattr__(self, name):
        return getattr(self._real_connection, name)
