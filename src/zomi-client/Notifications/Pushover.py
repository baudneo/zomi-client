import datetime
import logging
import pickle
import time
from enum import Enum, IntEnum
from pathlib import Path
from pickle import loads as pickle_loads, dump as pickle_dump
from typing import List, NoReturn, Union, Optional, Tuple, Dict

import numpy as np
import requests
import urllib3.exceptions
from pydantic import BaseModel, Field, AnyUrl

from ..main import get_global_config
from ..Models.config import GlobalConfig
from ..Log import CLIENT_LOGGER_NAME
from ..Notifications import CoolDownBase

logger = logging.getLogger(CLIENT_LOGGER_NAME)
g: Optional[GlobalConfig] = None


class Optionals(BaseModel):
    """Supplemental options for the pushover notification"""

    cache_write: bool = False
    # ("image.jpg", open("your_image.jpg", "rb"), "image/jpeg")
    attachment: Tuple = Field(default_factory=tuple)
    receipts: List = Field(default_factory=list)


class Limits(BaseModel):
    limit: int = Field(default=10000)
    remaining: int = Field(default=10000)
    reset: Optional[float] = Field(default=None)


class Priorities(IntEnum):
    LOWEST = lowest = -2
    LOW = low = -1
    NORMAL = normal = 0
    HIGH = high = 1
    EMERGENCY = emergency = 2


class PushoverURLs(BaseModel):
    """In the form of Tuple(URL: str, METHOD: str)"""

    base: Tuple[AnyUrl, str] = Field(("https://api.pushover.net/1/", ""))
    messages: Tuple[str, str] = Field(("messages.json", "post"))
    # sounds returns sound: description key: vals
    sounds: Tuple[str, str] = Field(("sounds.json", "get"))
    receipts: Tuple[str, str] = Field(("receipts/(your receipt).json", "get"))
    limits: Tuple[str, str] = Field(("apps/limits.json", "get"))
    validate_user: Tuple[str, str] = Field(("users/validate.json", "post"))


class RequestData(BaseModel):
    token: str = None
    user: str = None
    message: str = Field(default=None, max_length=1024)
    device: str = None
    # 1=on 0=off - only html OR monospace - not both
    html: int = 0
    monospace: int = 0
    priority: Optional[Priorities] = Field(Priorities.NORMAL)
    sound: dict = Field(default_factory=dict)
    timestamp: float = 0.0
    title: str = Field(default=None, max_length=250)
    url: str = Field(default=None, max_length=512)
    url_title: str = Field(default=None, max_length=100)
    # For Emergency (2) priority
    retry: int = Field(default=30)
    expire: int = Field(default=3600)
    # URL that Pushover will send a request to
    callback: str = None
    # ID to poll for status of emergency message
    # Did anyone interact with the message yet?

    # validators
    # @root_validator
    # def validate_keys(cls, values):
    #     print(f"in root validator RequestData: {values} ")
    #     tkn = values["token"]
    #     usr = values["user"]
    #     msg = values["message"]
    #
    #     if not tkn:
    #         raise ValueError("token is required")
    #     if len(tkn.strip()) != 30:
    #         raise ValueError("token MUST be 30 characters long")
    #     if not usr:
    #         raise ValueError("user is required")
    #     if len(usr.strip()) != 30:
    #         raise ValueError("user MUST be 30 characters long")
    #     if not msg:
    #         raise ValueError("message is required")
    #     if len(msg) > 1024:
    #         raise ValueError("message must be 1024 characters or less")
    #     # Optional, do length check
    #     title = values["title"]
    #     if title and len(title) > 250:
    #         raise ValueError("title must be 250 characters or less")
    #     url = values["url"]
    #     if url and len(url) > 512:
    #         raise ValueError("url must be 512 characters or less")
    #     url_title = values["url_title"]
    #     if url_title and len(url_title) > 100:
    #         raise ValueError("url_title must be 100 characters or less")
    #     device = values["device"]
    #     if device and len(device) > 25:
    #         raise ValueError("device must be 25 characters or less")
    #     priority = values["priority"]
    #     if priority:
    #         if priority not in range(-2, 3):
    #             raise ValueError("priority must be between -2 and 2")
    #         if priority == 2:
    #             retry = values["retry"]
    #             if retry and retry < 30:
    #                 raise ValueError("retry must be at least 30 seconds")
    #             expire = values["expire"]
    #             if expire and expire > 10800:
    #                 raise ValueError("expire must be at most 10800 seconds (3 Hours)")
    #     return values

    # @root_validator
    # def mono_or_html_only(cls, values):
    #     if values["monospace"] and values["html"]:
    #         match = re.search(r"<[^>]*>", values["message"])
    #         if match:
    #             logger.warning(
    #                 f"monospace and html cannot be used together, HTML tags "
    #                 f"detected in message body, ignoring monospace"
    #             )
    #             values["monospace"] = 0
    #         else:
    #             logger.warning(
    #                 f"monospace and html cannot be used together, NO HTML tags "
    #                 f"detected in message body, ignoring html"
    #             )
    #             values["html"] = 0
    #         return values


class Pushover(CoolDownBase):
    _request_data: RequestData = RequestData()
    _urls: PushoverURLs = PushoverURLs()
    _optionals: Optionals = Optionals()
    _limits: Limits = Limits()
    _push_auth: str = ""
    _data_dir_str: str = "push/pushover"


    def __init__(self):
        """Create a Pushover object to interact with the pushover service"""
        global g
        g = get_global_config()
        self.config = g.config.notifications.pushover
        self.lp: str = "PushOver:"
        self.data_dir = g.config.system.variable_data_path / self._data_dir_str
        self.data_dir.mkdir(parents=True, exist_ok=True)
        super().__init__()

    @property
    def image(self):
        return self.optionals.attachment

    @image.setter
    def image(self, img: np.ndarray):
        lp = "pushover::image::set::"
        import cv2
        is_succ, img = cv2.imencode(".jpg", img)
        img = img.tobytes()
        if not is_succ:
            logger.warning(
                f"{lp} cv2 failed to encode frame to an image"
            )
            raise Exception(
                "cv2 failed to encode frame to an image"
            )
        self.optionals.attachment = (
                    "objdetect.jpg",
                    img,
                    "image/jpeg",
                )

    def pickle(self, action: str = 'r', data: Optional[float] = None):
        """Pickle read/write the timestamp of each monitor's last successful pushover notification"""
        import os
        lp: str = "pushover::pickle::"
        action = action.casefold().strip()
        var_path = g.config.system.variable_data_path
        file = var_path / f"pushover_m{g.mid}.pkl"
        if action == "r":
            lp = f"{lp}read::"
            if file.exists():
                try:
                    time_since_sent = pickle_loads(file.read_bytes())
                except FileNotFoundError:
                    logger.debug(
                        f"{lp} FileNotFound - no time of last successful push found for monitor {g.mid}",
                    )
                    return
                except EOFError:
                    logger.debug(
                        f"{lp} empty file found for monitor {g.mid}, going to remove '{file}'",
                    )
                    try:
                        file.unlink(missing_ok=True)
                    except Exception as e:
                        logger.error(f"{lp} could not delete: {e}")
                except PermissionError as err:
                    logger.error(
                        f"{lp} PermissionError - could not read file: {file} -- {err}"
                    )
                except Exception as e:
                    logger.error(f"{lp} error while loading pickled data - Exception: {e}")
                else:
                    logger.debug(f"{lp} loaded data from '{file}'")
                    return time_since_sent
            else:
                logger.debug(f"{lp} no file found for monitor {g.mid}: '{file}'")

        elif action == "w":
            lp = f"{lp}write::"
            if data:
                if isinstance(data, float):
                    try:
                        file.touch(exist_ok=True, mode=0o640)
                        file.write_bytes(pickle.dumps(data))
                        logger.debug(
                            f"{lp} LAST successful push sent at {datetime.datetime.fromtimestamp(data)} ({data}) to '{file}'"
                        )
                    except Exception as e:
                        logger.error(
                            f"{lp} error writing to '{file}', time since last successful push sent not recorded: {e}"
                        )
                else:
                    logger.error(f"{lp} data is not a float: {type(data)}")
            else:
                logger.warning(f"{lp} no data to write to '{file}'")
        else:
            logger.warning(
                f"{lp} the action supplied: '{action}' is unknown only 'r|R' or 'w|W' are supported"
            )

    @property
    def urls(self) -> PushoverURLs:
        return self._urls

    @urls.setter
    def urls(self, options: dict):
        for opt, val_ in options.items():
            if opt in self._urls.__dict__:
                setattr(self._urls, opt, val_)

    @property
    def request_data(self):
        return self._request_data

    # create setter and deleter for options
    @request_data.setter
    def request_data(self, options: Union[dict, RequestData]):
        if isinstance(options, dict):
            self._request_data = RequestData(**options)
        elif isinstance(options, RequestData):
            self._request_data = options

        # for opt, val_ in options.items():
        #     if opt in self._request_data.__dict__:
        #         setattr(self._request_data, opt, val_)

    @property
    def optionals(self):
        return self._optionals

    @optionals.setter
    def optionals(self, options: dict):
        for opt, val_ in options.items():
            if opt in self._optionals.__dict__:
                setattr(self._optionals, opt, val_)

    @property
    def limits(self):
        """Return the limits object"""
        return self._limits

    @limits.setter
    def limits(self, options: Dict):
        for opt, val_ in options.items():
            if opt in self._limits.__dict__:
                setattr(self._limits, opt, val_)

    def parse_sounds(self, labels: List[str]) -> NoReturn:
        lp: str = "pushover sounds:"
        groups: Dict = g.config.label_groups
        weights: Dict = {}  # self.config. g.config.get("push_sounds_weights", PUSHOVER_SOUND_WEIGHTS)
        configured_sounds: Dict = self.config.sounds
        label_in_group: bool = False
        group_name: str
        logger.debug(f"DBG => PUSHOVER SOUND PARSING")
        if 'default' in configured_sounds:
            self.request_data.sound = configured_sounds.pop("default")
        # priority is in descending order
        for label in labels:
            # If the label is in a group, change label name to group name
            if groups:
                for group_name, group_value in groups.items():
                    group_value: Union[str, list]
                    if group_value:
                        group_value = group_value.strip().split(",")
                    if label in group_value:
                        logger.debug(f"DBG => pushover, {label} is in group named {group_name}")
                        label_in_group = True
            # if the label is in sounds
            if label in configured_sounds:
                logger.debug(f"DBG => {label} is in sounds")
                # if label_in_group and group_name in configured_sounds:
                #     logger.debug(f"DBG => USING {group_name} SOUND")
                #     self.request_data.sound = configured_sounds[group_name]
                # Check if there is a current sound already set
                if self.request_data.sound is not None:
                    logger.debug(f"DBG => current sound is {self._request_data.sound}, CHECKING WEIGHTS --> NOT IMPLEMENTED")
                    # Check to see about weights
                else:
                    logger.debug(
                        f"DEBUG <>>> self._request_data.sound is None, setting {configured_sounds[label]} as sound..."
                    )
                self._request_data.sound = configured_sounds[label]
            else:
                logger.debug(f"DEBUG <>>> {label} is not in {configured_sounds = }")
        logger.debug(f"{lp} end of method, result: {self._request_data.sound = }")

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Send the request to the Pushover API, same args and kwargs as :class:`requests.request`
        :param url: URL to send the request to
        :param method: HTTP method to use
        :param kwargs: Same as requests.request
        """
        return requests.request(method=method, url=url, **kwargs)

    def send(self):
        """Send Pushover notification"""
        lp: str = "pushover::send::"
        if not self.check_cooldown(g.mid):
            return
        logger.debug(f"{lp} {self.request_data = }")
        if self.request_data.priority == Priorities.EMERGENCY:
            logger.debug(f"{lp} emergency priority, setting retry and expire")

        sfx, method = self.urls.messages
        url = f"{self.urls.base[0]}{sfx}"
        logger.debug(f"{lp} {url = } -- {method = }")
        # Check cooldown


        try:
            data = self.request_data.dict()
            r = self.request(
                method,
                url,
                data=data,
                files={
                    "attachment": self.optionals.attachment
                }
            )
            r.raise_for_status()
        except urllib3.exceptions.ConnectionError as conn_exc:
            logger.error(
                f"{lp} there was a problem connecting to the pushover API -> {conn_exc}"
            )
            return
        except urllib3.exceptions.NewConnectionError as new_conn_exc:
            logger.error(f"{lp} check your internet connection! -> {new_conn_exc}")
            return
        except requests.exceptions.HTTPError as http_exc:
            http_exc: requests.exceptions.HTTPError
            status_code: int = http_exc.response.status_code
            if status_code == 429:
                logger.warning(f"{lp} you have exceeded the message limit or you are being rate limited! (429)")
            elif status_code == 401:
                logger.error(f"{lp} your pushover credentials are invalid! (401)")
            elif status_code == 500:
                logger.error(f"{lp} the pushover API is having issues! (500)")
            #      requests.exceptions.HTTPError: 413 Client Error: Request Entity Too Large for url: https://api.pushover.net/1/messages.json
            elif status_code == 413:
                logger.error(f"{lp} the message is too long! [Possible improper image encoding?] (413)")
            else:
                headers = http_exc.response.headers
                json_ = http_exc.response.json()
                text_ = http_exc.response.text
                logger.debug(f"DEBUG <>>> FAILED!!! {status_code = } || {headers = } || {json_ = } || {text_ = }")
                
        except Exception as exc:
            logger.error(f"{lp} there was a problem sending the message BROAD EXCEPTION -> {exc}")
            raise exc
        else:
            status_code: int = r.status_code
            headers = r.headers
            json_ = r.json()
            text_ = r.text
            self.limits.limit = headers.get("X-Limit-App-Limit")
            self.limits.remaining = headers.get("X-Limit-App-Remaining")
            self.limits.reset = headers.get("X-Limit-App-Reset")
            r_status = json_.get("status")
            r_request = json_.get("request")
            r_user = json_.get("user")
            r_errors = json_.get("errors")
            logger.debug(
                f"{lp} message Limit: {self.limits.limit} - remaining: {self.limits.remaining} "
                f"- reset counter Epoch: {self.limits.reset} "
                f"({datetime.datetime.fromtimestamp(float(self.limits.reset))})"
            )
            if r_status == 1:
                self.write_cooldown(g.mid)
                # self.pickle("w", time.time())
            else:
                logger.error(
                    f"{lp} pushover replied with FAILED: {json_}"
                )


def do_po_emerg(labels: List[str], *args, **kwargs) -> bool:
    po: Optional[Pushover] = None
    if len(args):
        po = args[0]
    send_push = False
    lp: str = "pushover:emerg:"
    emerg_mons: Union[str, set] = g.config.get("push_emerg_mons")
    if emerg_mons:
        emerg_mons = set(str(emerg_mons).strip().split(","))
        emerg_mons = {int(x) for x in emerg_mons}
        if g.mid in emerg_mons:
            proceed = True
            emerg_labels = g.config.get("push_emerg_labels")
            if emerg_labels:
                emerg_labels = set(str(emerg_labels).strip().split(","))
                if not any(
                        w in emerg_labels
                        for w in labels
                ):
                    logger.debug(
                        f"You have specified emergency labels ({emerg_labels}) that are not in the "
                        f"detected objects, not sending an emergency alert..."
                    )
                    proceed = False
                else:
                    logger.debug(
                        f"{lp} emergency labels ({emerg_labels}) match detected objects, "
                        f"sending emergency alert..."
                    )
            else:
                logger.debug(
                    f"DEBUG <>>> NO emergency labels configured!"
                )
            if proceed:
                logger.debug(
                    f"DEBUG!>>> EVALUATING DATEPARSER for emergency notification"
                )
                import dateparser

                def time_in_range(
                        start: Union[datetime.datetime, float],
                        end: Union[datetime.datetime, float],
                        current_: Optional[Union[datetime.datetime, float]] = None,
                ):
                    """Returns whether current is in the range [start, end]"""
                    if not current_:
                        current_ = datetime.datetime.now()
                    logger.debug(
                        f"DEBUG <>>> time in range: Checking CURRENT: {current_} to supplied "
                        f"timerange START: {start} - END: {end}"
                    )
                    return start <= current_ <= end

                tz = g.config.get("push_emerg_tz", {})
                if tz:
                    tz = {"TIMEZONE": tz}
                    logger.debug(f"{lp} converting to TimeZone: {tz}")
                # strip whitespace and convert the str into a list using a comma as the delimiter
                # convert to a set to remove duplicates and then use set comprehension to ensure all int
                emerg_retry = int(g.config.get("push_emerg_retry", 120))
                emerg_expire = int(
                    g.config.get("push_emerg_expire", 3600)
                )
                emerg_time_noon2mid = g.config.get(
                    "push_emerg_n2m", "11:00:00"
                )
                emerg_time_mid2noon = g.config.get(
                    "push_emerg_m2n", "06:00:00"
                )

                current = datetime.datetime.now()

                before_midnight = dateparser.parse("Today at 23:59:59")
                after_midnight = dateparser.parse("Today at 00:00:01")
                noon: datetime = dateparser.parse("Today at 12:00:00")
                # For logging purposes
                emerg_time_noon2mid = dateparser.parse(
                    emerg_time_noon2mid, settings=tz
                )
                emerg_time_mid2noon = dateparser.parse(
                    emerg_time_mid2noon, settings=tz
                )
                logger.debug(
                    f"DEBUG <>>> {current = } | {before_midnight = } "
                    f"| {after_midnight = } | {noon = }"
                )
                logger.debug(
                    f"DEBUG <>>> {emerg_time_noon2mid = } -- {emerg_time_mid2noon = }"
                )
                emerg_in_time: bool = False
                if before_midnight > current > noon:
                    logger.debug(
                        f"DEBUG <>>> currently between noon and midnight"
                    )
                    # between noon and midnight
                    emerg_in_time = time_in_range(
                        emerg_time_noon2mid,
                        before_midnight,
                    )
                elif after_midnight > current < noon:
                    logger.debug(
                        f"DEBUG <>>> currently between midnight and noon"
                    )
                    # after midnight, before noon
                    emerg_in_time = time_in_range(
                        after_midnight,
                        emerg_time_mid2noon,
                    )
                else:
                    logger.debug(
                        f"DEBUG <>>> WEIRD TIME ISSUE? -- currently {datetime.datetime.now() = }"
                    )

                if not emerg_in_time:
                    logger.debug(
                        f"{lp} it is currently not within the specified time range for "
                        f"sending an emergency notification"
                    )
                    # send a regular priority notification
                else:
                    logger.info(
                        f"{lp} sending pushover emergency notification..."
                    )
                    send_push = True
                    po.request_data.priority = 2
                    po.request_data.retry = emerg_retry
                    po.request_data.expire = emerg_expire
        else:
            logger.debug(
                f"{lp} the current monitor {g.mid} is not in the list of monitors that "
                f"receive emergency alerts ({emerg_mons})"
            )
    else:
        logger.debug(f"DEBUG <>>> {emerg_mons = } -- is not configured")
    return send_push
