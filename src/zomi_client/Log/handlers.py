import logging
import logging.handlers
from typing import Optional


class BufferedLogHandler(logging.handlers.BufferingHandler):
    def __init__(self, capacity=10000):
        logging.handlers.BufferingHandler.__init__(self, capacity)

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        return False

    def flush(self, *args, **kwargs):
        """Override flush to do nothing unless a file_handler kwarg is supplied."""
        file_handler: Optional[logging.FileHandler] = kwargs.get("file_handler")
        if len(self.buffer) > 0:
            if file_handler is not None and isinstance(file_handler, logging.FileHandler):
                self.acquire()
                # flush buffer to file handler
                try:
                    for record in self.buffer:
                        fh_record = file_handler.format(record)
                        file_handler.acquire()
                        try:
                            file_handler.stream.write(fh_record)
                            file_handler.stream.write(file_handler.terminator)
                            file_handler.flush()
                        finally:
                            file_handler.release()
                    self.buffer.clear()
                finally:
                    self.release()

    def flush2(self, *args, **kwargs):
        """
        Only print to stdout if there is no file handler to flush to AND file logging is enabled

        UPDATE: need to flush to stdout
        """
        from ..main import get_global_config

        fh_enabled = g.config.logging.file.enabled
        if fh_enabled:
            from . import CLIENT_LOGGER_NAME

            has_fh = False
            handlers = logging.getLogger(CLIENT_LOGGER_NAME).handlers
            for _handler in handlers:
                if isinstance(_handler, logging.FileHandler):
                    has_fh = True
            g = get_global_config()

            if not has_fh:
                msg = "No file handler to flush to, printing to stdout. This should help with errors on startup."
                print(msg)
                # flush to stdout
                for record in self.buffer:
                    print(record.getMessage())
                print(msg)
                self.buffer.clear()
