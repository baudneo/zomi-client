import logging
import re
import inspect
from typing import Optional, Dict, Tuple

from pydantic import FieldValidationInfo

from ..Log import CLIENT_LOGGER_NAME

logger = logging.getLogger(CLIENT_LOGGER_NAME)


def validate_percentage_or_pixels(v, field: FieldValidationInfo):
    # get func name programmatically
    _name_ = inspect.currentframe().f_code.co_name
    if v:
        re_match = re.match(r"(0*?1?\.?\d*\.?\d*?)(%|px)?$", str(v), re.IGNORECASE)
        if re_match:
            try:
                starts_with: Optional[re.Match] = None
                type_of = ""
                if re_match.group(1):
                    starts_with = re.search(
                        r"(0*\.|1\.)?(\d*\.?\d*?)(%|px)?$",
                        re_match.group(1),
                        re.IGNORECASE,
                    )
                    # logger.debug(
                    #     f"{_name_}:: '{field.name}' checking starts_with(): {starts_with.groups()}"
                    # )
                    if re_match.group(2) == "%":
                        # Explicit %
                        # logger.debug(f"{_name_}:: '{field.name}' is explicit %")
                        type_of = "Percentage"
                        v = float(re_match.group(1)) / 100.0
                    elif re_match.group(2) == "px":
                        # Explicit px
                        # logger.debug(f"{_name_}:: '{field.name}' is explicit px")
                        type_of = "Pixel"
                        v = int(re_match.group(1))
                    elif starts_with and not starts_with.group(1):
                        """
                        'total_max_area' is valid REGEX re_match: ('1.0', None)
                        'total_max_area' checking starts_with(): ('', '1.0', None)
                        """
                        # there is no '%' or 'px' at end and the string does not start with 0*., ., or 1.
                        # consider it a pixel input (basically an int)
                        # logger.debug(
                        #     f"{_name_}:: '{field.name}' :: there is no '%' or 'px' at end and the string "
                        #     f"does not start with 0*., ., or 1. - CONVERTING TO INT AS PIXEL VALUE"
                        # )
                        type_of = "Pixel"
                        v = int(float(re_match.group(1)))
                    else:
                        # String starts with 0*., . or 1. treat as a float type percentile
                        # logger.debug(
                        #     f"{_name_}:: '{field.name}' :: String starts with [0*., ., 1.] treat as "
                        #     f"a float type percentile"
                        # )
                        type_of = "Percentage"
                        v = float(re_match.group(1))
                    # logger.debug(f"{type_of} value detected for {field.name} ({v})")
            except TypeError or ValueError as e:
                logger.warning(
                    f"{field.field_name} value: '{v}' could not be converted to a FLOAT! -> {e} "
                )
                v = 1
            except Exception as e:
                logger.warning(
                    f"{field.field_name} value: '{v}' could not be converted -> {e} "
                )
                v = 1
        else:
            logger.warning(f"{field.field_name} value: '{v}' malformed!")
            v = 1
    return v


def validate_resolution(v, **kwargs):
    _RESOLUTION_STRINGS: Dict[str, Tuple[int, int]] = {
        # pixel resolution string to tuple, feed it .casefold().strip()'d string's
        # flip the tuple to be (height, width) instead of (width, height)
        "4kuhd": (2160, 3840),
        "uhd": (2160, 3840),
        "4k": (2160, 4096),
        "6MP": (2048, 3072),
        "5MP": (1944, 2592),
        "4MP": (1520, 2688),
        "3MP": (1536, 2048),
        "2MP": (1200, 1600),
        "1MP": (1024, 1280),
        "1440p": (1440, 2560),
        "2k": (1080, 2048),
        "1080p": (1080, 1920),
        "960p": (960, 1280),
        "720p": (720, 1280),
        "fullpal": (576, 720),
        "fullntsc": (480, 720),
        "pal": (576, 704),
        "ntsc": (480, 704),
        "4cif": (480, 704),
        "2cif": (240, 704),
        "cif": (240, 352),
        "qcif": (120, 176),
        "480p": (480, 854),
        "360p": (360, 640),
        "240p": (240, 426),
        "144p": (144, 256),
    }
    logger.debug(f"Validating Monitor Zone resolution: {v}")
    if not v:
        logger.warning("No resolution provided for monitor zone, will not be able to scale "
                       "zone Polygon if resolution changes")
    elif isinstance(v, str):
        v = v.casefold().strip()
        if v in _RESOLUTION_STRINGS:
            v = _RESOLUTION_STRINGS[v]
        elif v not in _RESOLUTION_STRINGS:
            # check for a valid resolution string
            import re
            # WxH
            if re.match(r"^\d+x\d+$", v):
                v = tuple(int(x) for x in v.split("x"))
            # W*H
            elif re.match(r"^\d+\*\d+$", v):
                v = tuple(int(x) for x in v.split("*"))
            # W,H
            elif re.match(r"^\d+,\d+$", v):
                v = tuple(int(x) for x in v.split(","))
            else:
                logger.warning(
                    f"Invalid resolution string: {v}. Valid strings are: W*H WxH W,H OR "
                    f"{', '.join(_RESOLUTION_STRINGS)}"
                )
    return v


def validate_points(v, field, **kwargs):
    if v:
        orig = str(v)
        if not isinstance(v, (str, list)):
            raise TypeError(
                f"'{field.field_name}' Can only be List or string! type={type(v)}"
            )
        elif isinstance(v, str):
            v = [tuple(map(int, x.strip().split(","))) for x in v.split(" ")]
        from shapely.geometry import Polygon

        try:
            Polygon(v)
        except Exception as exc:
            logger.warning(f"Zone points unable to form a valid Polygon: {exc}")
            raise TypeError(
                f"The polygon points [coordinates] supplied "
                f"are malformed! -> {orig}"
            )
        else:
            assert isinstance(v, list)

    return v


