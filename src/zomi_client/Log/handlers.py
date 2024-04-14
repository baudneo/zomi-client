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
        Override flush to do nothing unless a file_handler kwarg is supplied.

        UPDATE: need to flush to stdout


         """
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
            # elif file_handler is None:
            #     # flush to stdout
            #     for record in self.buffer:
            #         print(record.getMessage())
            #         print("PRINTED OUT DUE TO NO FILE HANDLER TO FLUSH TO!")
            #     self.buffer.clear()

