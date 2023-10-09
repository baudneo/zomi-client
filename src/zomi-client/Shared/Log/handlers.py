import logging
import logging.handlers
from typing import Optional


class BufferedLogHandler(logging.handlers.BufferingHandler):
    def __init__(self, capacity=10000):
        logging.handlers.BufferingHandler.__init__(self, capacity)

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        return False


    def flush(self, *args, **kwargs):
        """
        Override flush to do nothing unless a file_handler kwarg is supplied


         """
        # only flush if a file_handler is set
        file_handler: Optional[logging.FileHandler] = kwargs.get("file_handler")
        if len(self.buffer) > 0 and file_handler is not None and isinstance(file_handler, logging.FileHandler):
            self.acquire()
            # flush buffer to file handler
            try:
                for record in self.buffer:
                    record = file_handler.format(record)
                    file_handler.acquire()
                    try:
                        file_handler.stream.write(record)
                        file_handler.stream.write(file_handler.terminator)
                        file_handler.flush()
                    finally:
                        file_handler.release()
                self.buffer.clear()
            finally:
                self.release()
