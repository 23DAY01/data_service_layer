from mysql.connector.utils.performance import QueryTuner


class _CursorProxy:
    def __init__(self, real_cursor, conn_config):
        self._real_cursor = real_cursor
        use_tuning = conn_config.get('use_performance_tuning', True)
        self._tuner = QueryTuner(tuning_enabled=use_tuning)

    def execute(self, operation, params=None, multi=False):
        processed_params = self._tuner.process_parameters(operation, params)
        return self._real_cursor.execute(operation, processed_params, multi)

    def __getattr__(self, name):
        return getattr(self._real_cursor, name)
