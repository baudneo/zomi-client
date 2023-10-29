import logging

CLIENT_LOGGER_NAME: str = "ZoMi:c"
CLIENT_LOG_FORMAT = logging.Formatter(
    "'%(asctime)s.%(msecs)04d' %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
