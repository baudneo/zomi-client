import logging

from .handlers import BufferedLogHandler

CLIENT_LOGGER_NAME: str = "zomi:C"
# "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s"
fmt = "%(asctime)s.%(msecs)04d %(name)s %(levelname)s[%(module)s:%(lineno)d] -> %(message)s"
date_fmt = "%m/%d/%y %H:%M:%S"
CLIENT_LOG_FORMAT = logging.Formatter(
    fmt,
    date_fmt,
)
