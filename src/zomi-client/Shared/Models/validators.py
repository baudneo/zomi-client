import inspect
import logging
import re
from pathlib import Path
from typing import Union, Any, Optional, Dict

from pydantic import FieldValidationInfo

from ...Client.Log import CLIENT_LOGGER_NAME
from ...Server.Log import SERVER_LOGGER_NAME

# find loggers
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
if CLIENT_LOGGER_NAME in loggers:
    logger = logging.getLogger(CLIENT_LOGGER_NAME)
elif SERVER_LOGGER_NAME in loggers:
    logger = logging.getLogger(SERVER_LOGGER_NAME)
else:
    logger = logging.getLogger(CLIENT_LOGGER_NAME)

def validate_not_enabled(v, **kwargs):
    if v is None:
        v = False
    return v

def validate_enabled(v, **kwargs):
    if v is None:
        v = True
    return v


def validate_no_scheme_url(v, info: FieldValidationInfo):
    """Validate and transform a URL/IP string into a URL with a scheme"""
    _name_ = inspect.currentframe().f_code.co_name
    logger.debug(f"{_name_}:: Validating '{info.field_name}' -> {v}")
    if v:
        import re

        if re.match(r"^(http(s)?)://", v):
            pass
            logger.debug(f"'{info.field_name}' is valid with schema: {v}")
        else:
            logger.debug(
                f"No schema in '{info.field_name}', prepending http:// to make {info.field_name} a valid URL"
            )
            v = f"http://{v}"
    return v


def validate_octal(v, **kwargs):
    """Validate and transform octal string into an octal"""
    assert isinstance(v, str)
    if v:
        if re.match(r"^(0o[0-7]+)$", v):
            pass
        else:
            raise ValueError(f"Invalid octal string: {v}")
    return v


def validate_log_level(v, **kwargs):
    """Validate and transform log level string into a log level"""
    if v:
        assert isinstance(v, str)
        v = v.strip().upper()
        if re.match(r"^(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)$", v):
            if v == "WARN":
                v = "WARNING"
            elif v == "FATAL":
                v = "CRITICAL"
            if v == "INFO":
                v = logging.INFO
            elif v == "DEBUG":
                v = logging.DEBUG
            elif v == "WARNING":
                v = logging.WARNING
            elif v == "ERROR":
                v = logging.ERROR
            elif v == "CRITICAL":
                v = logging.CRITICAL
        else:
            raise ValueError(f"Invalid log level string: {v}")
    return v


def str2path(v: Union[str, Path, None], info: Optional[FieldValidationInfo] = None, **kwargs):
    """Convert a str to a Path object - pydantic validator

    Args:
        v (str|path|None): string to convert to a Path object
    Keyword Args:
        field (FieldValidationInfo): pydantic field object
    """
    if v:
        assert isinstance(v, (Path, str))
        v = Path(v)
        v.expanduser().resolve()
    return v


def validate_dir(v, info: Optional[FieldValidationInfo] = None):
    if v:
        v = str2path(v, info)
        assert v.exists(), f"Path [{v}] does not exist"
        assert v.is_dir(), f"Path [{v}] is not a directory"
    return v


def validate_file(v, info: Optional[FieldValidationInfo] = None):
    if v:
        v = str2path(v, info)
        assert v.exists(), f"Path [{v}] does not exist"
        assert v.is_file(), f"Path [{v}] is not a file"
    return v


def str2bool(v: Optional[Union[Any, bool]], **kwargs) -> Union[Any, bool]:
    """Convert a string to a boolean value, if possible.

    .. note::
        - The string is converted to all lower case before evaluation.
        - Strings that will return True -> ("yes", "true", "t", "y", "1", "on", "ok", "okay", "da").
        - Strings that will return False -> ("no", "false", "f", "n", "0", "off", "nyet").
        - None is converted to False.
        - A boolean is returned as-is.
    """
    if v is not None:
        true_ret = ("yes", "true", "t", "y", "1", "on", "ok", "okay", "da", "enabled")
        false_ret = ("no", "false", "f", "n", "0", "off", "nyet", "disabled")
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            v = str(v)
        if isinstance(v, str):
            if (normalized_v := str(v).lower().strip()) in true_ret:
                return True
            elif normalized_v in false_ret:
                pass
            else:
                return logger.warning(
                    f"str2bool: The value '{v}' (Type: {type(v)}) is not able to be parsed into a boolean operator"
                )
        else:
            return logger.warning(
                f"str2bool: The value '{v}' (Type: {type(v)}) is not able to be parsed into a boolean operator"
            )
    else:
        return None
    return False
