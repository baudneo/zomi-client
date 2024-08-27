from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import copy
import json
import logging
import logging.handlers
import os
import pickle
import re
import signal
import sys
import time
import warnings
from pathlib import Path
from shutil import which
from typing import Union, Dict, Optional, List, Any, Tuple, TYPE_CHECKING

from .Models.validators import str2bool

try:
    import cv2
except ImportError:
    cv2 = None
    msg = (
        "OpenCV is not installed. This is required for image processing. "
        "Please install OpenCV to enable image processing."
    )
    warnings.warn(
        msg,
        ImportWarning,
    )
    print(msg)
try:
    import numpy as np
    import yaml
    from pydantic import BaseModel, Field
    from shapely.geometry import Polygon
except ImportError as e:
    msg = (
        f"Some dependencies are not installed. Please install them to enable "
        f"all features. {e}"
    )
    warnings.warn(
        msg,
        ImportWarning,
    )
    print(msg)
    yaml = None
    np = None
    Polygon = None
    BaseModel = None
    Field = None
    raise e

from .Libs.Media.pipeline import (
    APIImagePipeLine,
    ZMSImagePipeLine,
)
from .Libs.API import ZMAPI
from .Models.utils import CFGHash, get_push_auth
from .Models.config import (
    ConfigFileModel,
    ServerRoute,
    OverRideMatchFilters,
    MonitorZones,
    MatchFilters,
    OverRideStaticObjects,
    OverRideObjectFilters,
    OverRideFaceFilters,
    OverRideAlprFilters,
    MatchStrategy,
    NotificationZMURLOptions,
    ClientEnvVars,
    GlobalConfig,
    LoggingSettings,
    Testing,
    DetectionResults,
    Result,
)

from .Log import CLIENT_LOGGER_NAME, CLIENT_LOG_FORMAT, BufferedLogHandler

if TYPE_CHECKING:
    from .Libs.DB import ZMDB
    from .Notifications.Pushover import Pushover
    from .Notifications.Gotify import Gotify
    from .Notifications.zmNinja import ZMNinja
    from .Notifications.MQTT import MQTT
    from .Notifications.ShellScript import ShellScriptNotification

ZM_INSTALLED: Optional[str] = which("zmpkg.pl")

logger = logging.getLogger(CLIENT_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

g: Optional[GlobalConfig] = None
LP: str = "Client:"


def set_logger(supplied_logger: logging.Logger) -> None:
    global logger
    if logger is not None:
        if isinstance(logger, logging.Logger):
            logger.info(
                f"{LP} CHANGING LOGGERS! Current: '{logger.name}' - Setting logger to {supplied_logger.name}"
            )
            logger = supplied_logger
            logger.info(f"{LP} logger has been changed to {logger.name}")


def create_logs() -> logging.Logger:
    global logger

    del logger
    logger = logging.getLogger(CLIENT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(CLIENT_LOG_FORMAT)
    buffered_log_handler = BufferedLogHandler()
    buffered_log_handler.setFormatter(CLIENT_LOG_FORMAT)
    logger.addHandler(console_handler)
    logger.addHandler(buffered_log_handler)
    return logger


async def init_logs(config: ConfigFileModel) -> None:
    """Initialize the logging system."""
    import getpass
    import grp

    sys_user: str = getpass.getuser()
    sys_gid: int = os.getgid()
    sys_group: str = grp.getgrgid(sys_gid).gr_name
    sys_uid: int = os.getuid()

    cfg: LoggingSettings = config.logging
    root_level = cfg.level
    logger.debug(f"Setting root logger level to {logging._levelToName[root_level]}")
    logger.setLevel(root_level)

    if cfg.console.enabled is False:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                logger.info("Removing console log output!")
                logger.removeHandler(h)

    if cfg.file.enabled:
        if cfg.file.file_name:
            _filename = cfg.file.file_name
        else:
            _filename = f"{cfg.file.filename_prefix}_m{g.mid}.log"
        abs_logfile = cfg.file.path / _filename
        try:
            if not abs_logfile.exists():
                logger.info(f"Creating log file [{abs_logfile}]")
                abs_logfile.touch(exist_ok=True, mode=0o644)
            else:
                with abs_logfile.open("a") as f:
                    pass
        except PermissionError:
            logger.warning(
                f"Logging to file disabled due to permissions"
                f" - No write access to '{abs_logfile.as_posix()}' for user: "
                f"{sys_uid} [{sys_user}] group: {sys_gid} [{sys_group}]"
            )
        else:
            # ZM /var/log/zm is handled by logrotate
            file_handler = logging.FileHandler(abs_logfile.as_posix(), mode="a")
            file_handler.setFormatter(CLIENT_LOG_FORMAT)
            if g.config.logging.file.level:
                logger.debug(
                    f"File logger level CONFIGURED AS {g.config.logging.file.level}"
                )
                # logger.debug(f"Setting file log level to '{logging._levelToName[g.config.logging.file.level]}'")
                file_handler.setLevel(g.config.logging.file.level)
            logger.addHandler(file_handler)
            # get the buffered handler and call flush with file_handler as a kwarg
            # this will flush the buffer to the file handler
            for h in logger.handlers:
                if isinstance(h, BufferedLogHandler):
                    logger.debug(
                        f"Flushing buffered log handler to file --- {file_handler=}"
                    )
                    h.flush(file_handler=file_handler)
                    # Close the buffered handler
                    h.close()
                    break
            logger.debug(
                f"Logging to file '{abs_logfile}' with user: "
                f"{sys_uid} [{sys_user}] group: {sys_gid} [{sys_group}]"
            )
    if cfg.syslog.enabled:
        # enable syslog logging
        syslog_handler = logging.handlers.SysLogHandler(
            address=cfg.syslog.address,
        )
        syslog_handler.setFormatter(CLIENT_LOG_FORMAT)
        if cfg.syslog.level:
            logger.debug(
                f"Syslog logger level CONFIGURED AS {logging._levelToName[cfg.syslog.level]}"
            )
            syslog_handler.setLevel(cfg.syslog.level)
        logger.addHandler(syslog_handler)
        logger.debug(f"Logging to syslog at {cfg.syslog.address}")

    logger.info(f"Logging initialized...")


def parse_client_config_file(
    cfg_file: Path, template: Optional[Union[ConfigFileModel, Any]] = None
) -> Optional[ConfigFileModel]:
    """Parse the YAML configuration file."""
    if template is None:
        template = ConfigFileModel
    cfg: Dict = {}
    _start = time.time()
    raw_config = cfg_file.read_text()

    try:
        cfg = yaml.safe_load(raw_config)
    except yaml.YAMLError as yaml_exc:
        logger.error(f"Error parsing the YAML configuration file!")
        raise yaml_exc
    except PermissionError as perm_exc:
        logger.error(f"Error reading the YAML configuration file!")
        raise perm_exc

    substitutions = cfg.get("substitutions", {})
    testing = cfg.get("testing", {})
    testing = Testing(**testing)
    if testing.enabled:
        logger.info(f"|----- TESTING IS ENABLED! -----|")
        if testing.substitutions:
            logger.info(f"Overriding config:substitutions WITH testing:substitutions")
            substitutions = testing.substitutions

    logger.debug(f"Replacing ${{VARS}} in config:substitutions")
    for x in range(3):
        substitutions = _replace_vars(str(substitutions), substitutions)

    # secrets
    if inc_file := substitutions.get("IncludeFile"):
        inc_file = Path(inc_file)
        logger.debug(f"PARSING IncludeFile: {inc_file.as_posix()}")
        if inc_file.is_file():
            inc_vars = yaml.safe_load(inc_file.read_text())
            if "client" in inc_vars:
                inc_vars = inc_vars.get("client", {})
                logger.debug(
                    f"Loaded {len(inc_vars)} substitution from IncludeFile {inc_file}"
                )
                # check for duplicates
                for k in inc_vars:
                    if k in substitutions:
                        logger.warning(
                            f"Duplicate substitution variable '{k}' in IncludeFile {inc_file} - "
                            f"IncludeFile overrides config file"
                        )

                substitutions.update(inc_vars)
            else:
                logger.warning(
                    f"IncludeFile [{inc_file}] does not have a 'client' section - skipping"
                )
        else:
            logger.warning(f"IncludeFile {inc_file} is not a file!")
    logger.debug(f"Replacing ${{VARS}} in config")
    cfg = _replace_vars(raw_config, substitutions)
    logger.debug(
        f"perf:: Config file loaded and validated in {time.time() - _start:.5f} seconds"
    )

    return template(**cfg)


def _replace_vars(search_str: str, var_pool: Dict) -> Dict:
    """Replace variables in a string.


    Args:
        search_str (str): String to search for variables '${VAR_NAME}'.
        var_pool (Dict): Dictionary of variables used to replace.

    """
    import re
    import yaml

    if var_list := re.findall(r"\$\{(\w+)\}", search_str):
        # $ remove duplicates
        var_list = list(set(var_list))
        logger.debug(f"Found the following substitution variables: {var_list}")
        # substitute variables
        _known_vars = []
        _unknown_vars = []
        for var in var_list:
            if var in var_pool:
                # logger.debug(
                #     f"substitution variable '{var}' IS IN THE POOL! VALUE: "
                #     f"{var_pool[var]} [{type(var_pool[var])}]"
                # )
                _known_vars.append(var)
                value = var_pool[var]
                if value is None:
                    value = ""
                elif value is True:
                    value = "yes"
                elif value is False:
                    value = "no"
                search_str = search_str.replace(f"${{{var}}}", str(value))
            else:
                _unknown_vars.append(var)
        if _unknown_vars:
            logger.warning(
                f"The following variables have no configured substitution value: {_unknown_vars}"
            )
        if _known_vars:
            logger.debug(
                f"The following variables have been substituted: {_known_vars}"
            )
    else:
        logger.debug(f"No substitution variables found.")

    return yaml.safe_load(search_str)


def get_global_config() -> GlobalConfig:
    if g is None:
        return create_global_config()
    return g


def set_global_config(config: GlobalConfig) -> None:
    global g

    g = config


def create_global_config() -> GlobalConfig:
    """Create the global config object."""
    global g

    g = GlobalConfig()
    return g


class StaticObjects(BaseModel):
    labels: Optional[List[str]] = Field(default_factory=list)
    confidence: Optional[List[float]] = Field(default_factory=list)
    bbox: Optional[List[List[int]]] = Field(default_factory=list)
    filename: Optional[Path] = None

    def pickle(
        self,
        labels: Optional[List[str]] = None,
        confs: Optional[List] = None,
        bboxs: Optional[List] = None,
        write: bool = False,
    ) -> Optional[Tuple[List[str], List, List]]:
        """Use the pickle module to read or write the static objects to a file.

        :param write: save the data to a file
        :param bboxs: list of bounding boxes (Required for write)
        :param confs: list of confidence scores (Required for write)
        :param labels: list of labels (Required for write)
        """
        lp: str = f"static_objects:{'write' if write else 'read'}:"

        variable_data_path = g.config.system.variable_data_path
        filename = self.filename
        if not write:
            logger.debug(
                f"{lp} trying to load previous detection results from file: '{filename}'"
            )
            if filename.exists():
                _data = None
                try:
                    with filename.open("rb") as f:
                        labels = pickle.load(f)
                        confs = pickle.load(f)
                        bboxs = pickle.load(f)
                except FileNotFoundError:
                    logger.debug(
                        f"{lp}  no history data file found for monitor '{g.mid}'"
                    )
                except EOFError:
                    logger.debug(f"{lp}  empty file found for monitor '{g.mid}'")
                    logger.debug(f"{lp}  going to remove '{filename}'")
                    try:
                        filename.unlink()
                    except Exception as e:
                        logger.error(f"{lp}  could not delete: {e}")
                except Exception as e:
                    logger.error(f"{lp} error: {e}")
                else:
                    logger.debug(
                        f"{lp} read file success! returning results: {labels}, {confs}, {bboxs}"
                    )
            else:
                logger.warning(f"{lp} no history data file found for monitor '{g.mid}'")
        else:
            # _past_match = (saved_label, saved_conf, saved_bbox)
            propagations = g.static_objects.get("propagate")
            _l = list(labels)
            _c = list(confs)
            _b = list(bboxs)
            if propagations:
                logger.debug(f"{lp} ADDING propagations: {propagations}")
                for prop in propagations:
                    _l.append(prop[0])
                    _c.append(prop[1])
                    _b.append(prop[2])
            else:
                logger.debug(f"{lp} NO propagations to add")
            try:
                filename.touch(exist_ok=True, mode=0o640)
                with filename.open("wb") as f:
                    pickle.dump(_l, f)
                    pickle.dump(_c, f)
                    pickle.dump(_b, f)
                    logger.debug(
                        f"{lp} saved_event: {g.eid} RESULTS to file: '{filename}' ::: {labels}, {confs}, {bboxs}",
                    )
            except Exception as e:
                logger.error(
                    f"{lp}  error writing to '{filename}' past detections not recorded, err msg -> {e}"
                )
        self.labels = labels
        self.confidence = confs
        self.bbox = bboxs
        return self.labels, self.confidence, self.bbox


class Notifications:
    mqtt: Optional[MQTT] = None
    zmninja: Optional[ZMNinja] = None
    gotify: Optional[Gotify] = None
    pushover: Optional[Pushover] = None
    shell_script: Optional[ShellScriptNotification] = None
    webhook = None

    def __init__(self):
        from .Notifications.Pushover import Pushover
        from .Notifications.Gotify import Gotify
        from .Notifications.zmNinja import ZMNinja
        from .Notifications.MQTT import MQTT
        from .Notifications.ShellScript import ShellScriptNotification

        config = g.config.notifications
        if config.zmninja.enabled:
            self.zmninja = ZMNinja()
        if config.gotify.enabled:
            self.gotify = Gotify()
            # Gotify config allows for overriding portal url
            if config.gotify.portal:
                _portal = str(config.gotify.portal)
            else:
                _portal = str(g.api.portal_base_url)
            # get link user auth
            has_https = True
            if not re.compile(r"https://").match(_portal):
                has_https = False
            if config.gotify.clickable_link:
                self.gotify._push_auth = get_push_auth(
                    g.api, config.gotify.link_user, config.gotify.link_pass, has_https
                )

        if config.pushover.enabled:
            # get link user auth
            has_https = True
            if not re.compile(r"^https://").match(g.api.portal_base_url):
                has_https = False
            self.pushover = Pushover()
            if config.pushover.clickable_link:
                self.pushover._push_auth = get_push_auth(
                    g.api,
                    config.pushover.link_user,
                    config.pushover.link_pass,
                    has_https,
                )

        if config.shell_script.enabled:
            self.shell_script = ShellScriptNotification()
        if config.mqtt.enabled:
            self.mqtt = MQTT()
            self.mqtt.connect()


class ZMClient:
    config_file: Union[str, Path]
    config_hash: CFGHash
    raw_config: str
    parsed_cfg: Dict
    config: ConfigFileModel
    api: ZMAPI
    db: ZMDB
    routes: List[ServerRoute]
    mid: int
    eid: int
    image_pipeline: Union[APIImagePipeLine, ZMSImagePipeLine]
    _comb: Dict

    @staticmethod
    def is_live_event(is_live: bool):
        if is_live is True:
            if g:
                g.past_event = False
        else:
            if g:
                g.past_event = True

    def signal_handler_int(self, *args, **kwargs):
        logger.info(f"Received signal: SIGINT, cleaning connections up and exiting")
        self.signal_handler_clean_up(*args, **kwargs)

    def signal_handler_term(self, *args, **kwargs):
        logger.info(f"Received signal: SIGTERM, cleaning connections up and exiting")
        self.signal_handler_clean_up(*args, **kwargs)

    def signal_handler_clean_up(self, *args, **kwargs):
        self.db.clean_up()
        self.api.session.close()
        asyncio.get_event_loop().create_task(self.api.async_session.close())
        asyncio.get_event_loop().stop()

    async def clean_up(self):
        logger.debug(f"closing API and DB connection(s)")
        # self.image_pipeline.exit()
        await self.api.clean_up()
        self.db.clean_up()

    def __init__(self, global_config: Optional[GlobalConfig] = None):
        """
        Initialize the ZoneMinder Client
        """
        global logger

        if not ZM_INSTALLED:
            _msg = "ZoneMinder is not installed, the client requires to be installed on a ZoneMinder host!"
            logger.error(_msg)
            raise RuntimeError(_msg)
        lp = f"{LP}init:"

        # setup async signal catcher
        loop = asyncio.get_event_loop()
        if not logger:
            logger = create_logs()
        signals = ("SIGINT", "SIGTERM")
        logger.debug(
            f"{lp} registering signal handler for {' ,'.join(signals).rstrip(',')}"
        )

        loop.add_signal_handler(
            getattr(signal, "SIGINT"),
            self.signal_handler_int,
        )
        loop.add_signal_handler(
            getattr(signal, "SIGTERM"),
            self.signal_handler_term,
        )

        if global_config:
            logger.debug(f"{lp} Using supplied global config")
            set_global_config(global_config)
        global g
        g = get_global_config()
        # DO env vars
        g.Environment = ClientEnvVars()
        self.zones: Dict = {}
        self.zone_polygons: List[Polygon] = []
        self.zone_filters: Dict = {}
        self.filtered_labels: Dict = {}
        self.static_objects = StaticObjects()
        self.notifications: Optional[Notifications] = None
        self.config = get_global_config().config
        futures: List[concurrent.futures.Future] = []
        _hash: concurrent.futures.Future

        # _hash_input = CFGHash(config_file=g.config_file)
        # loop = asyncio.get_event_loop()
        # loop.create_task(self._sort_routes())
        # loop.create_task(self._init_api())
        # loop.create_task(self._init_db())
        with concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="init", max_workers=g.config.system.thread_workers
        ) as executor:
            # _hash = executor.submit(lambda: _hash_input.compute())
            futures.append(executor.submit(self._init_db))
            futures.append(executor.submit(self._init_api))
            for future in concurrent.futures.as_completed(futures):
                future.result()
        # self.config_hash = _hash.result()

    async def _get_db_data(self, eid: int):
        """Get data from the database"""
        # return mid, mon_name, mon_post, mon_pre, mon_fps, reason, event_path, \
        #        notes, width, height, color, ring_buffer
        mid: int = 0
        (
            mid,
            g.mon_name,
            g.mon_post,
            g.mon_pre,
            g.mon_fps,
            g.event_cause,
            g.event_path,
            g.notes,
            g.mon_width,
            g.mon_height,
            g.mon_colorspace,
            g.mon_image_buffer_count,
        ) = self.db.grab_all(eid)
        if (mid and g.mid) and (g.mid != mid):
            logger.debug(
                f"{LP} CLI supplied monitor ID ({g.mid}) INCORRECT! Changed to: {mid}"
            )
        if mid:
            if not g.mid or g.mid != mid:
                logger.debug(f"{LP} Setting GLOBAL (Current: {g.mid}) monitor ID to: {mid}")
                g.mid = mid

    def _init_db(self):
        from .Libs.DB import ZMDB

        self.db = ZMDB()
        logger.debug(f"DB initialized")

    def _init_api(self):
        g.api = self.api = ZMAPI(g.config.zoneminder)
        logger.debug("API initialized")
        self.notifications = Notifications()

    @staticmethod
    async def convert_to_cv2(image: Union[np.ndarray, bytes]):
        # convert the numpy image to OpenCV format
        lp = "convert_to_cv2::"
        if isinstance(image, bytes):
            # logger.debug(f"{lp} image is in bytes, converting to cv2 numpy array")
            # image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.imdecode(
                np.asarray(bytearray(image), dtype=np.uint8), cv2.IMREAD_COLOR
            )

        return image

    def combine_filters(
        self,
        filters_1: Union[Dict, MatchFilters, OverRideMatchFilters],
        filters_2: Union[Dict, OverRideMatchFilters],
    ):
        lp: str = "combine filters:"
        # logger.debug(f"{lp} BASE filters [type: {type(filters_1)}]: {filters_1}")
        # logger.debug(f"{lp} OVERRIDE filters [type: {type(filters_2)}]: {filters_2}")
        if isinstance(filters_1, (MatchFilters, OverRideMatchFilters)):
            # logger.debug(f"{lp} filters_1 is a MatchFilters object, converting to dict")
            output_filters: Dict = filters_1.model_dump()
        elif isinstance(filters_1, dict):
            output_filters = filters_1
        else:
            raise TypeError("filters_1 must be a dict or (OverRide)MatchFilters object")
        _base_obj_label: Dict = output_filters["object"]["labels"]
        # logger.debug(f"{lp} BASE object.labels: {_base_obj_label}")

        if isinstance(filters_2, (MatchFilters, OverRideMatchFilters)):
            # logger.debug(
            #     f"{lp} filters_2 is a {type(filters_2)} object, converting to dict"
            # )
            override_filters: Dict = filters_2.model_dump()
        elif isinstance(filters_2, dict):
            override_filters = filters_2
        elif filters_2 is None:
            override_filters = {}
        else:
            raise TypeError("filters_2 must be a dict or OverRideMatchFilters object")
        # logger.debug(f"{lp} OVERRIDE object.labels: {_override_obj_label}")
        # if _base_obj_label is None:
        #     _base_obj_label = {}
        if output_filters:
            if override_filters:
                _override_obj_label: Dict = override_filters["object"]["labels"]
                if _override_obj_label:
                    for label, filter_data in _override_obj_label.items():
                        if output_filters["object"]["labels"] is None:
                            output_filters["object"]["labels"] = {}

                        if _base_obj_label and label in _base_obj_label:
                            for k, v in filter_data.items():
                                if (
                                    v is not None
                                    and v
                                    != output_filters["object"]["labels"][label][k]
                                ):
                                    # logger.debug(
                                    #     f"{lp} Overriding BASE filter 'object':'labels':'{label}':'{k}' with "
                                    #     f"Monitor {g.mid} OVERRIDE filter VALUE '{v}'"
                                    # )
                                    output_filters["object"]["labels"][label][k] = v
                        else:
                            # logger.debug(
                            #     f"{lp} Adding Monitor {g.mid} OVERRIDE filter 'object':'labels'"
                            #     f":'{label}' with VALUE '{filter_data}'"
                            # )
                            output_filters["object"]["labels"][label] = filter_data

                for filter_type, filter_data in override_filters.items():
                    if filter_data is not None:
                        for k, v in filter_data.items():
                            if k == "labels":
                                # Handled in the first loop
                                continue
                            if v is not None and v != output_filters[filter_type][k]:
                                output_filters[filter_type][k] = v
                                # logger.debug(
                                #     f"{lp} Overriding BASE filter '{filter_type}':'{k}' with Monitor {g.mid} "
                                #     f"OVERRIDE filter VALUE '{v}'"
                                # )
            # logger.debug(f"{lp} Final combined output => {output_filters}")
        if not output_filters["object"]["labels"]:
            output_filters["object"]["labels"] = None
        self._comb_filters = output_filters
        return self._comb_filters

    async def detect(self, eid: Optional[int] = None, mid: Optional[int] = None):
        """Detect objects in an event

        Args:
            eid (Optional[int]): Event ID. Required for API event image pulling method.
            mid (Optional[int]): Monitor ID. Required for SHM or ZMU image pulling method.
        """
        global g
        lp = _lp = "detect::"
        final_detections: dict = {}
        detections: dict = {}
        matched_l, matched_c, matched_b, matched_e = [], [], [], []
        matched_model_names = ""
        matched_frame_id = ""
        matched_detection_types = ""
        matched_processor = ""
        matched_frame_img = np.ndarray([])
        image_name: Optional[Union[int, str]] = None
        _start = time.time()
        global g
        strategy: MatchStrategy = g.config.matching.strategy
        img_pull_method = self.config.zoneminder.pull_method
        if eid:
            g.eid = eid
            logger.info(
                f"{lp} Running detection for event {eid}, obtaining monitor info using ZoneMinder DB..."
            )

            await self._get_db_data(eid)

            # Check that the cause from db has "Motion", "ONVIF" or "Trigger" in it
            if g.config.detection_settings.motion_only is True:
                cause = self.db.cause_from_eid(eid)
                _cont = True
                if cause:
                    if (
                        cause.find("Motion") == -1
                        and cause.find("Trigger") == -1
                        and cause.find("ONVIF") == -1
                        # and cause.find("Forced") == -1
                    ):
                        _cont = False
                    else:
                        _cont = True
                else:
                    _cont = False

                if not _cont:
                    logger.info(
                        f"{lp} Event {eid} is not a motion, ONVIF or trigger event, skipping detection"
                    )
                    return None

            await self.db.get_all_event_data(eid)
        elif not eid and mid:
            logger.info(
                f"{lp} Running detection for monitor {mid}, image pull method {img_pull_method}"
            )
            g.mid = mid

        await init_logs(g.config)
        await self.db.import_zones()
        # Sets absolute file path for pickled data file.
        self.static_objects.filename = (
            g.config.system.variable_data_path / f"misc/.static-objects_m{g.mid}.pkl"
        )

        # init Image Pipeline
        logger.debug(f"{lp} Initializing Image Pipeline...")
        if img_pull_method and img_pull_method.zms.enabled is True:
            logger.debug(f"{lp} Using ZMS CGI for image source")
            self.image_pipeline = ZMSImagePipeLine(img_pull_method.zms)
        elif img_pull_method.api and img_pull_method.api.enabled is True:
            logger.debug(f"{lp} Using ZM API for image source")
            self.image_pipeline = APIImagePipeLine(img_pull_method.api)

        models: Optional[Dict] = None
        self.static_objects.pickle()
        create_default_zone: bool = True
        # If no ML zones are set up in the config file by the user, create a default zone of the whole image
        if "create_default_full_image_zone" in self.config.monitors:
            temp_var = self.config.monitors.pop("create_default_full_image_zone", None)
            if temp_var is not None:
                create_default_zone = str2bool(temp_var)
        # check for zones and assign them
        if g.mid in self.config.monitors:
            if self.config.monitors[g.mid].zones:
                self.zones = self.config.monitors[g.mid].zones
            # check if the monitor config has models defined, if so, override the global config
            if self.config.monitors[g.mid].models:
                logger.debug(
                    f"{lp} Monitor {g.mid} has models configured, overriding global models"
                )
                models = self.config.monitors[g.mid].models
        else:
            # There is absolutely no ML config defined for this monitor, ML will never run.
            logger.critical(f"{lp} Monitor {g.mid} not found in monitors: section!")
        # make sure we have some models defeined to run on.
        if not models:
            if self.config.detection_settings.models:
                logger.debug(
                    f"{lp} Monitor {g.mid} has NO config entry for MODELS, using global "
                    f"models from detection_settings:"
                )
                models = self.config.detection_settings.models
            else:
                logger.debug(
                    f"{lp} Monitor {g.mid} has NO config entry for MODELS and global "
                    f"models from detection_settings is empty, defaulting to 'yolov4'"
                )
                models = {"yolov4": {}}
        _start_detections = time.time()
        base_filters = g.config.matching.filters
        if g.mid in g.config.monitors:
            monitor_filters = g.config.monitors.get(g.mid).filters
            # logger.debug(f"{lp} Combining GLOBAL filters with Monitor {g.mid} filters")
            combined_filters = self.combine_filters(base_filters, monitor_filters)
        else:
            combined_filters = base_filters

        if not self.zones and create_default_zone is True:
            logger.debug(f"{lp} No zones found, adding full image with base filters")
            self.zones["!zomi-client!_full_image"] = MonitorZones.model_construct(
                points=[
                    (0, 0),
                    (g.mon_height, 0),
                    (g.mon_height, g.mon_width),
                    (0, g.mon_width),
                ],
                resolution=(int(g.mon_width), int(g.mon_height)),
                object_confirm=False,
                static_objects=OverRideStaticObjects(),
                filters=OverRideMatchFilters(),
            )

        if self.zones is not None:
            # build each zones filters as they won't change, check points and resolution for scaling
            zones = self.zones.copy()
            mon_res = (g.mon_width, g.mon_height)
            for zone_name, zone_data in zones.items():
                cp_fltrs = copy.deepcopy(combined_filters)
                self.zone_filters[zone_name] = self.combine_filters(
                    cp_fltrs, self.zones[zone_name].filters
                )
                if zone_data.enabled is False:
                    continue
                if not zone_data.points:
                    continue
                zone_points = zone_data.points
                zone_resolution = zone_data.resolution
                if zone_resolution != mon_res:
                    logger.warning(
                        f"{_lp} Zone '{zone_name}' has a resolution of '{zone_resolution}'"
                        f" which is different from the monitor resolution of {mon_res}! "
                        f"Attempting to scale zone to match monitor resolution..."
                    )

                    xfact = mon_res[1] / zone_resolution[1] if zone_resolution[1] != 0 else 1.0
                    yfact = mon_res[0] / zone_resolution[0] if zone_resolution[0] != 0 else 1.0

                    logger.debug(
                        f"{_lp} Rescaling polygons using x_factor: {xfact} and y_factor: {yfact} "
                        f"(mon_res={mon_res}, zone_res={zone_resolution})"
                    )

                    zone_points = [(int(x * xfact), int(y * yfact)) for x, y in zone_points]

                    logger.debug(f"{_lp} Zone '{zone_name}' points adjusted to: {zone_points}")

                    self.zones[zone_name].points = zone_points
            del zones
        image: Union[bytes, np.ndarray, None]
        mlapi_cfg = g.config.mlapi
        matched_l, matched_c, matched_b = [], [], []
        from .Libs.API.mlapi import MLAPI
        mlapi = MLAPI(mlapi_cfg)


        image_loop = 0
        # todo: add re-login logic if the token is rejected.
        # Use an async generator to get images from the image pipeline
        break_out: bool = False
        image_start = time.time()
        async for image, image_name in self.image_pipeline.generate_image():
            image_loop += 1
            if break_out is True:
                logger.debug(
                    f"perf:{LP} IMAGE LOOP #{image_loop} ({image_name}) took {time.time() - image_start:.5f} s"
                )
                break

            if strategy == MatchStrategy.first and matched_l:
                logger.debug(
                    f"Strategy is 'first' and there is a filtered match, breaking out of IMAGE loop {image_loop}"
                )
                break_out = True
                continue

            if image is None:
                # None is returned if no image was returned by API
                logger.warning(f"{lp} No image returned! trying again...")
                continue
            if image is False:
                # False is returned if the image stream has been exhausted
                logger.warning(f"{lp} Image stream has been exhausted!")
                break
            assert isinstance(
                image, bytes
            ), "Image is not bytes after getting from pipeline"
            image: bytes

            results: Optional[List[DetectionResults]] = None
            reply: Optional[Dict[str, Any]] = None
            # JSONify, images and hints, base64 encode the image and wrap it in a list
            images = json.dumps([base64.b64encode(image).decode("utf-8")])
            # hints are model names we want run
            hints = json.dumps([model for model in models.keys() if model])
            reply = await mlapi.inference(images, hints)

            if image_name not in final_detections:
                logger.debug(f"Adding {image_name=} to final_detections")
                final_detections[str(image_name)] = []
            if image_name not in self.filtered_labels:
                logger.debug(f"Adding {image_name=} to self.filtered_labels")
                self.filtered_labels[str(image_name)] = []

            if reply:
                results = []
                logger.debug(f"DBG>>> {reply = }")
                image = await self.convert_to_cv2(image)
                assert isinstance(
                    image, np.ndarray
                ), "Image is not np.ndarray after converting from bytes"
                image: np.ndarray

                for img_results in reply:
                    for result_ in img_results:
                        results.append(DetectionResults(**result_))

                filter_start = time.time()
                result_loop = 0
                for result in results:
                    result_loop += 1
                    if break_out is True:
                        continue
                    if strategy == MatchStrategy.first and matched_l:
                        logger.debug(
                            f"Strategy is 'first' and there is a filtered match, "
                            f"breaking out of RESULT loop {result_loop}"
                        )
                        break_out = True
                        continue

                    logger.debug(
                        f"{LP} starting to process results from model: '{result.name}'"
                    )

                    if result.success is True:
                        l = len(result.results)
                        logger.debug(
                            f"There {'are' if l > 1 else 'is'} {l} UNFILTERED {'results' if l > 1 else 'result'} "
                            f"from model: {result.name} for image '{image_name}'"
                        )
                        filtered_result = await self.filter_detections(
                            result, image_name
                        )


                        # todo: fix the strategy matching logic. first-match, first-frame
                        strategy: MatchStrategy = g.config.matching.strategy
                        if filtered_result.success is True:
                            final_label = []
                            final_confidence = []
                            final_bbox = []
                            for _res in filtered_result.results:
                                final_label.append(_res.label)
                                final_confidence.append(_res.confidence)
                                final_bbox.append(_res.bounding_box)

                            if (
                                (strategy == MatchStrategy.first)
                                or (
                                    (strategy == MatchStrategy.most)
                                    and (len(final_label) > len(matched_l))
                                )
                                or (
                                    (strategy == MatchStrategy.most)
                                    and (len(final_label) == len(matched_l))
                                    and (sum(matched_c) < sum(final_confidence))
                                )
                                # or (
                                # (frame_strategy == "most_models")
                                # and (len(item["detection_types"]) > len(matched_detection_types))
                                # )
                                #         or (
                                #         (strategy == "most_models")
                                #         and (len(item["detection_types"]) == len(matched_detection_types))
                                #         and (sum(matched_c) < sum(item["confidences"]))
                                # )
                                or (
                                    (strategy == MatchStrategy.most_unique)
                                    and (len(set(final_label)) > len(set(matched_l)))
                                )
                                or (
                                    # tiebreaker using sum of confidences
                                    (strategy == MatchStrategy.most_unique)
                                    and (len(set(final_label)) == len(set(matched_l)))
                                    and (sum(matched_c) < sum(final_confidence))
                                )
                            ):
                                logger.debug(
                                    f"\n\nFOUND A BETTER MATCH [{strategy=}] THAN model: {matched_model_names}"
                                    f" image name: {matched_frame_id}: LABELS: {matched_l} with "
                                    f" model: {result.name} image name: {image_name} ||| "
                                    f"LABELS: {final_label}\n\n"
                                )

                                matched_l = final_label
                                matched_model_names = result.name
                                matched_c = final_confidence
                                matched_frame_id = image_name
                                matched_detection_types = result.type
                                matched_b = final_bbox
                                matched_processor = result.processor
                                # FIXME: Is filtered out objects really needed in the results?
                                if str(image_name) in self.filtered_labels:
                                    matched_e = self.filtered_labels[str(image_name)]
                                else:
                                    matched_e = []
                                matched_frame_img = image.copy()

                            final_detections[str(image_name)].append(filtered_result)

                        logger.debug(
                            f"perf:{LP} Filtering for model: '{result.name}' image name: {image_name} took "
                            f"{time.time() - filter_start:.5f} seconds"
                        )

                    else:
                        logger.warning(f"{LP} detection result was not successful, not filtering")

            img_time = time.time() - image_start
            img_msg = f"perf:{LP} IMAGE #{image_loop} took {img_time:.5f} seconds"
            if strategy == MatchStrategy.first and matched_l:
                logger.debug(
                    f"Strategy is 'first' and there is a filtered match, breaking out of IMAGE loop {image_loop}"
                )
                logger.debug(
                    img_msg
                )
                break

            if (not g.past_event and img_time < 1.0):
                if image_loop >= self.image_pipeline.max_frames:
                    logger.debug(f"{img_msg}")
                else:
                    remaining = 1.0 - img_time
                    img_msg = (
                        f"{img_msg} (live target: 1 FPS), sleeping for {remaining:.5f}"
                    )
                    logger.debug(f"{img_msg}")
                    await asyncio.sleep(remaining)
                    logger.debug("DBG>>> END OF SLEEP")
            else:
                logger.debug(f"{img_msg}")
            image_start = time.time()

        monitor_perf_end = time.time() - _start
        if matched_l:
            matched = {
                "labels": matched_l,
                "model_names": matched_model_names,
                "confidences": matched_c,
                "frame_id": matched_frame_id,
                "detection_types": matched_detection_types,
                "bounding_boxes": matched_b,
                "processor": matched_processor,
                "frame_img": matched_frame_img,
                "filtered_bounding_boxes": matched_e,
            }
            logger.debug(
                f"based on strategy of {strategy}, BEST MATCH IS {matched['labels']} from frame ID: {matched['frame_id']}"
            )

            if not g.past_event:
                await self.post_process(matched)
                self.static_objects.pickle(
                    labels=matched_l, confs=matched_c, bboxs=matched_b, write=True
                )
            else:
                logger.info(
                    f"{LP} This is a past event, not post processing or writing static_object data"
                )

            if "frame_img" in matched:
                matched.pop("frame_img")

        else:
            matched = {}

        logger.debug(
            f"perf:{LP} Camera: '{g.mon_name}' (ID: {g.mid}) :: TOTAL detections took "
            f"{monitor_perf_end:.5f} seconds"
        )
        return matched

    async def filter_detections(
        self,
        result: DetectionResults,
        image_name: str,
    ) -> DetectionResults:
        """Filter detections"""
        lp: str = "filter detections::"

        final_results = []
        filtered_results = await self._filter(result, image_name=image_name)
        if filtered_results:
            for _fr in filtered_results:
                if _fr not in final_results:
                    final_results.append(_fr)
                    logger.debug(f"{lp} {_fr} not in final results, appending")
                else:
                    pass

        result.success = False if not final_results else True
        result.results = final_results

        return result

    @staticmethod
    def _bbox2points(bbox: List) -> list[tuple[tuple[Any, Any], tuple[Any, Any]]]:
        """Convert bounding box coords to a Polygon acceptable input for shapely."""
        orig_ = list(bbox)
        if isinstance(bbox[0], int):
            it = iter(bbox)
            bbox = list(zip(it, it))
        bbox.insert(1, (bbox[1][0], bbox[0][1]))
        bbox.insert(3, (bbox[0][0], bbox[2][1]))
        # logger.debug(f"convert bbox:: {orig_} to Polygon points: {bbox}")
        return bbox

    async def _filter(
        self,
        result: DetectionResults,
        image_name: str = None,
        *args,
        **kwargs,
    ) -> List[Result]:
        """Filter detections using a loop within a loop, outer loop is filter by object label, inner loop is to filter by zone."""

        global_filters = g.config.matching.filters
        mon_cfg = g.config.monitors.get(g.mid)
        mon_filters = mon_cfg.filters
        zones = self.zones.copy()
        zone_filters = self.zone_filters
        typed_label_filters = {}
        final_filters = None
        final_filter_src: str = "<NOT_SET>"
        base_filters = None
        # strategy: MatchStrategy = g.config.matching.strategy
        type_ = result.type
        model_name = result.name
        ret_results = []
        processor = result.processor
        found_match: bool = False
        label, confidence, bbox, color = None, None, None, None
        _zn_tot = len(zones)
        _lbl_tot = len(result.results)
        idx = 0
        i = 0
        zone_name: str
        zone_data: MonitorZones
        _result: Result
        type_filter: Union[
            OverRideObjectFilters,
            OverRideFaceFilters,
            OverRideAlprFilters,
            None,
        ] = None
        _lp = f"_filter:{image_name}:'{model_name}'::{type_}::"

        def filter_out(lbl, cnf, box):
            """Filter out detections"""
            self.filtered_labels[str(image_name)].append(
                (
                    lbl,
                    cnf,
                    box,
                )
            )

        # Outer Loop (results)
        for _result in result.results:
            label, confidence, bbox, color = (
                _result.label,
                _result.confidence,
                _result.bounding_box,
                _result.color
            )
            i += 1
            _lp = f"fltr:{model_name}:'{label}' {i}/{_lbl_tot}:"

            idx = 0
            found_match = False
            skip_imported_zones: bool = False

            if mon_cfg:
                if mon_cfg.skip_imported_zones is not None:
                    skip_imported_zones = mon_cfg.skip_imported_zones

            # Inner Loop (zones)
            for zone_name, zone_data in zones.items():
                idx += 1
                __lp = f"{_lp}zone {idx}/{_zn_tot}::"
                if skip_imported_zones is True and zone_data.imported is True:
                    logger.debug(
                        f"{__lp} Zone '{zone_name}' is imported and 'skip_imported_zones' is configured, skipping..."
                    )
                    continue
                if zone_data.enabled is False:
                    logger.debug(
                        f"{__lp} Zone '{zone_name}' is disabled by config file..."
                    )
                    continue
                if not zone_data.points:
                    logger.warning(
                        f"{__lp} Zone '{zone_name}' has no points! Did you rename a Zone in ZM"
                        f" or forget to add points? SKIPPING..."
                    )
                    continue

                zone_points = zone_data.points
                zone_polygon = Polygon(zone_points)
                _data = {"name": zone_name, "points": zone_polygon}
                if _data not in self.zone_polygons:
                    self.zone_polygons.append(_data)
                bbox_polygon = Polygon(self._bbox2points(bbox))

                if bbox_polygon.intersects(zone_polygon):
                    logger.debug(
                        f"{__lp} inside of Zone '{zone_name}' @ {list(zip(*zone_polygon.exterior.coords.xy))[:-1]}"
                    )
                    if zone_name in zone_filters and zone_filters[zone_name]:
                        logger.debug(f"{__lp} zone '{zone_name}' has filters defined")
                        final_filters = zone_filters[zone_name]
                        final_filter_src = "<ZONE>"
                    else:
                        logger.debug(
                            f"{__lp} zone '{zone_name}' has NO filters, using COMBINED global+monitor filters"
                        )
                        final_filters = self._comb_filters
                        final_filter_src = "<COMBINED>"

                    # check if there are per-label filters set, if so override the final filters
                    if isinstance(final_filters, dict) and isinstance(
                        final_filters.get(type_), dict
                    ):
                        if ((typed_label_filters := final_filters[type_]["labels"]) and (label in typed_label_filters)
                                and typed_label_filters[label]):
                            logger.debug(
                                f"{__lp} '{label}' IS IN {type_.upper()} per-label filters"
                            )

                            for k, v in typed_label_filters[label].items():
                                if (
                                    v is not None
                                    and final_filters[type_][k] != v
                                ):
                                    logger.debug(
                                        f"{__lp} Overriding {type_}:'{k}' [{final_filters[type_][k]}] "
                                        f"with per-label {type_}:labels:{label}:{k}  [VALUE={v}]"
                                    )
                                    final_filters[type_][k] = v
                        else:
                            logger.debug(f"{__lp} '{label}' is NOT IN {type_.upper()} per-label filters")

                        final_filters = self.construct_filter(final_filters)
                        self.zone_filters[zone_name] = final_filters
                        # logger.debug(
                        #     f"\n\n'AFTER' SAVING TO ZONE FILTERS: {self.zone_filters=} \n\n"
                        # )


                    if type_ == "object":
                        type_filter = final_filters.object
                    elif type_ == "face":
                        type_filter = final_filters.face
                    elif type_ == "alpr":
                        type_filter = final_filters.alpr
                    # logger.debug(f"DBG>>>\n\n{global_filters = }\n\n{mon_filters = }\n\n"
                    #              f"{zone_filters = }\n\n{final_filters = }\n\n{type_filter = }\n\n")
                    # logger.debug(f"DBG>>>\n\n{self.zone_filters.get(zone_name) = }\n\n"
                    #              f"{mon_cfg.zones.get(zone_name).filters = }\n\n")

                    #
                    # Start filtering
                    #
                    pattern = type_filter.pattern
                    lp = f"{__lp}pattern match::"
                    if match := pattern.match(label):
                        # When using .* in the pattern, the match.groups() will return an empty tuple
                        if label in match.groups() or pattern.pattern == ".*":
                            logger.debug(
                                f"{lp} matched ReGex pattern [{pattern.pattern}] ALLOWING..."
                            )
                            if type_ == "face":
                                # logger.debug(
                                #     f"DBG>> This model is typed as {type_} is not OBJECT, skipping non face filters like min_conf, total_max_area, etc."
                                # )
                                found_match = True
                                break

                            lp = f"{__lp}min conf::"
                            if confidence >= type_filter.min_conf:
                                logger.debug(
                                    f"{lp} {confidence} IS GREATER THAN OR EQUAL TO "
                                    f"min_conf={type_filter.min_conf}, ALLOWING..."
                                )
                                if type_ == "alpr":
                                    # logger.debug(
                                    #     f"DBG>> This model is typed as {type_} is not OBJECT, skipping non alpr filters like total_max_area, etc."
                                    # )
                                    found_match = True
                                    break
                                w, h = g.mon_width, g.mon_height
                                max_object_area_of_image: Optional[
                                    Union[float, int]
                                ] = None
                                min_object_area_of_image: Optional[
                                    Union[float, int]
                                ] = None
                                max_object_area_of_zone: Optional[
                                    Union[float, int]
                                ] = None
                                min_object_area_of_zone: Optional[
                                    Union[float, int]
                                ] = None

                                # check total max area
                                lp = f"{__lp}total max area:"
                                if tma := type_filter.total_max_area:
                                    if isinstance(tma, float):
                                        if tma >= 1.0:
                                            tma = 1.0
                                            max_object_area_of_image = h * w
                                        else:
                                            max_object_area_of_image = tma * (h * w)
                                            logger.debug(
                                                f"{lp} converted {tma * 100.00}% of {w}*{h}->{w * h:.2f} to "
                                                f"{max_object_area_of_image:.2f} pixels",
                                            )

                                        if max_object_area_of_image > (h * w):
                                            max_object_area_of_image = h * w
                                    elif isinstance(tma, int):
                                        max_object_area_of_image = tma
                                    else:
                                        logger.warning(
                                            f"{lp} Unknown type for total_max_area, defaulting to PIXELS "
                                            f"h*w of image ({h * w})"
                                        )
                                        max_object_area_of_image = h * w
                                    perc_covered = (bbox_polygon.area / max_object_area_of_image) * 100.00
                                    if max_object_area_of_image:
                                        if bbox_polygon.area > max_object_area_of_image:
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} ({perc_covered}%) is larger than the "
                                                f"max allowed: {max_object_area_of_image:.2f},"
                                                f"\n\nREMOVING REMOVING REMOVING REMOVING REMOVING REMOVING...\n\n"
                                            )
                                            continue
                                        else:
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} ({perc_covered}%) is smaller than the "
                                                f"TOTAL (image w*h) max allowed: {max_object_area_of_image:.2f}"
                                                f", ALLOWING..."
                                            )
                                else:
                                    logger.debug(f"{lp} no total_max_area set")

                                # check total min area
                                lp = f"{__lp}total min area:"
                                if tmia := type_filter.total_min_area:
                                    if isinstance(tmia, float):
                                        if tmia > 1.0:
                                            tmia = 1.0
                                            min_object_area_of_image = h * w
                                        else:
                                            min_object_area_of_image = (
                                                tmia * zone_polygon.area
                                            )
                                            logger.debug(
                                                f"{lp} converted {tmia * 100.00}% of {w}*{h}->{w * h:.2f} to "
                                                f"{min_object_area_of_image:.2f} pixels",
                                            )

                                    elif isinstance(tmia, int):
                                        min_object_area_of_image = tmia
                                    else:
                                        logger.warning(
                                            f"{lp} Unknown type for total_min_area, defaulting to 1 PIXEL"
                                        )
                                        min_object_area_of_image = 1
                                    perc_covered = (bbox_polygon.area / min_object_area_of_image) * 100.00
                                    if min_object_area_of_image:
                                        if (
                                            bbox_polygon.area
                                            >= min_object_area_of_image
                                        ):
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} ({perc_covered}%) is LARGER THEN OR "
                                                f"EQUAL TO the TOTAL min allowed: {min_object_area_of_image:.2f}"
                                                f", ALLOWING..."
                                            )
                                        else:
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} ({perc_covered}%) is smaller then the "
                                                f"TOTAL min allowed: {min_object_area_of_image:.2f}, "
                                                f"\n\nREMOVING REMOVING REMOVING REMOVING REMOVING REMOVING...\n\n"
                                            )
                                            continue
                                else:
                                    logger.debug(f"{lp} no total_min_area set")

                                # check max area compared to zone
                                lp = f"{__lp}zone max area:"
                                if (max_area := type_filter.max_area) is not None:
                                    if isinstance(max_area, float):
                                        if max_area > 1.0:
                                            max_area = 1.0
                                            max_object_area_of_zone = zone_polygon.area
                                        else:
                                            max_object_area_of_zone = (
                                                max_area * zone_polygon.area
                                            )
                                            logger.debug(
                                                f"{lp} converted {max_area * 100.00}% of '{zone_name}'->"
                                                f"{zone_polygon.area:.2f} to {max_object_area_of_zone} pixels",
                                            )
                                        if max_object_area_of_zone > zone_polygon.area:
                                            max_object_area_of_zone = zone_polygon.area
                                    elif isinstance(max_area, int):
                                        max_object_area_of_zone = max_area
                                    else:
                                        logger.warning(
                                            f"{lp} Unknown type for max_area, defaulting to PIXELS "
                                            f"of zone [{zone_polygon.area}]"
                                        )
                                        max_object_area_of_zone = zone_polygon.area

                                    if max_object_area_of_zone:
                                        perc_covered = (bbox_polygon.intersection(zone_polygon).area / max_object_area_of_zone) * 100.00
                                        if (
                                            bbox_polygon.intersection(zone_polygon).area
                                            > max_object_area_of_zone
                                        ):
                                            logger.debug(
                                                f"{lp} BBOX AREA [{bbox_polygon.area:.2f} ({perc_covered}%)] is larger "
                                                f"than the max: {max_object_area_of_zone},"
                                                f"\n\nREMOVING REMOVING REMOVING REMOVING REMOVING REMOVING...\n\n"
                                            )
                                            continue
                                        else:
                                            logger.debug(
                                                f"{lp} '{label}' BBOX AREA [{bbox_polygon.area:.2f} ({perc_covered}%)] "
                                                f"is smaller than the max allowed: {repr(max_object_area_of_zone)}, "
                                                f"ALLOWING..."
                                            )
                                else:
                                    logger.debug(f"{lp} no max_area set")

                                # check min area compared to zone
                                lp = f"{__lp}zone min area:"
                                if (min_area := type_filter.min_area) is not None:
                                    if isinstance(min_area, float):
                                        if min_area > 1.0:
                                            min_area = 1.0
                                            min_object_area_of_zone = zone_polygon.area
                                        else:
                                            min_object_area_of_zone = (
                                                min_area * zone_polygon.area
                                            )
                                            logger.debug(
                                                f"{lp} converted {min_area * 100.00:.2f}% of '{zone_name}'->"
                                                f"{zone_polygon.area:.2f} to {min_object_area_of_zone} pixels"
                                            )
                                        if (
                                            min_object_area_of_zone
                                            > zone_polygon.area
                                        ):
                                            min_object_area_of_zone = zone_polygon.area
                                    elif isinstance(min_area, int):
                                        min_object_area_of_zone = min_area
                                    else:
                                        min_object_area_of_zone = 1
                                    perc_covered = (bbox_polygon.intersection(zone_polygon).area /
                                                    min_object_area_of_zone) * 100.00

                                    if min_object_area_of_zone:
                                        if (
                                            bbox_polygon.intersection(zone_polygon).area
                                            > min_object_area_of_zone
                                        ):
                                            logger.debug(
                                                f"{lp} '{label}' BBOX AREA [{bbox_polygon.area:.2f} ({perc_covered:.2f}%)] "
                                                f"is larger then the min allowed: {repr(min_object_area_of_zone)}, "
                                                f"ALLOWING..."
                                            )

                                        else:
                                            logger.debug(
                                                f"{lp} '{label}' BBOX AREA [{bbox_polygon.area:.2f} ({perc_covered:.2f}%)] "
                                                f"is smaller then the min allowed: {min_object_area_of_zone:.2f} ("
                                                f"{(bbox_polygon.area / min_object_area_of_zone)*100.00:.2f}%),"
                                                f"\n\nREMOVING REMOVING REMOVING REMOVING REMOVING REMOVING...\n\n"
                                            )
                                            continue
                                    else:
                                        logger.debug(f"{lp} no min_object_area_of_zone set")
                                else:
                                    logger.debug(f"{lp} no min_area set")

                                # color filtering
                                logger.debug(f"DBG>>> this is where color filtering should go: {color = } // "
                                             f"{g.config.detection_settings.color = }")

                                s_o: Optional[bool] = None
                                s_o_reason: Optional[str] = "<DFLT>"
                                s_o = g.config.matching.static_objects.enabled
                                if s_o is None:
                                    s_o = False
                                elif s_o in [True, False]:
                                    s_o_reason = "global"
                                zone_cfg: Optional[MonitorZones] = None
                                if mon_cfg and zone_name in mon_cfg.zones:
                                    zone_cfg = mon_cfg.zones[zone_name]

                                # Override with monitor filters than zone filters
                                if s_o is True:
                                    if mon_cfg and mon_cfg.static_objects:
                                        if mon_cfg.static_objects.enabled is False:
                                            s_o = False
                                            s_o_reason = "monitor"
                                elif s_o is False:
                                    if mon_cfg and mon_cfg.static_objects:
                                        if mon_cfg.static_objects.enabled is True:
                                            s_o = True
                                            s_o_reason = "monitor"
                                # zone filters override monitor filters
                                if s_o is False:
                                    if (
                                        zone_cfg
                                        and zone_cfg.static_objects.enabled is True
                                    ):
                                        s_o = True
                                        s_o_reason = "zone"
                                elif s_o is True:
                                    if (
                                        zone_cfg
                                        and zone_cfg.static_objects.enabled is False
                                    ):
                                        s_o = False
                                        s_o_reason = "zone"
                                if s_o is True:
                                    logger.debug(
                                        f"{__lp} 'static_objects' enabled (lvl: {s_o_reason}), checking for matches"
                                    )
                                    if self.check_for_static_objects(
                                        label, confidence, bbox_polygon, zone_name
                                    ):
                                        # success
                                        logger.debug(
                                            f"{__lp} SUCCESSFULLY PASSED the static object check"
                                        )

                                    else:
                                        logger.debug(
                                            f"{__lp} FAILED the static object check, continuing to next zone..."
                                        )
                                        # failed
                                        continue
                                else:
                                    logger.debug(
                                        f"{__lp} 'static_objects' disabled (lvl: {s_o_reason}), skipping check..."
                                    )
                                # !!!!!!!!!!!!!!!!!!!!
                                # End of all filters
                                # !!!!!!!!!!!!!!!!!!!!
                                found_match = True
                                break

                            else:
                                logger.debug(
                                    f"{lp} confidence={confidence} IS LESS THAN "
                                    f"min_confidence={type_filter.min_conf}, continuing to next zone..."
                                )
                                continue

                        else:
                            logger.debug(
                                f"{lp} NOT matched in RegEx pattern GROUPS [{pattern.pattern}], "
                                f"continuing to next zone..."
                            )
                            continue

                    else:
                        logger.debug(
                            f"{lp} NOT matched in RegEx pattern [{pattern.pattern}], continuing to next zone..."
                        )
                        continue
                else:
                    logger.debug(
                        f"{__lp} NOT in zone '{zone_name}', continuing to next zone..."
                    )
                    # log the bbox and zone coords
                    logger.debug(
                        f"{__lp} bbox: {list(zip(*bbox_polygon.exterior.coords.xy))[:-1]}"
                    )
                    logger.debug(
                        f"{__lp} zone: {list(zip(*zone_polygon.exterior.coords.xy))[:-1]}"
                    )

                # logger.debug(
                #     f"\n---------------------END OF ZONE LOOP # {idx} ---------------------"
                # )

            if found_match:
                logger.debug(f"{_lp} PASSED FILTERING")
                ret_results.append(
                    Result(
                        **{
                            "label": label,
                            "confidence": confidence,
                            "bounding_box": bbox,
                        }
                    )
                )

                if (strategy := g.config.matching.strategy) == MatchStrategy.first:
                    logger.debug(
                        f"{_lp} Match strategy: '{strategy}', breaking out of LABEL loop..."
                    )
                    break
            else:
                logger.debug(f"{_lp} FAILED FILTERING")
                filter_out(label, confidence, bbox)

            # logger.debug(
            #     f"\n---------------------END OF LABEL LOOP # {i} ---------------------"
            # )

        return ret_results

    def create_animations(self, label, confidence, bbox):
        """
        Create animations and save to disk
        :param label:
        :param confidence:
        :param bbox:
        :return:
        """
        lp = f"create_animations::"
        logger.debug(f"{lp} STARTED")

    def check_for_static_objects(
        self, current_label, current_confidence, current_bbox_polygon, zone_name
    ) -> bool:
        """Check for static objects in the frame
        :param current_label:
        :param current_confidence:
        :param current_bbox_polygon:
        """

        lp = f"check_for_static_objects::"
        logger.debug(f"{lp} STARTING...")
        aliases: Dict = g.config.label_groups
        _max_diff_area = g.config.matching.static_objects.difference
        _labels: Optional[List[str]] = self.static_objects.labels
        _confs: Optional[List[float]] = self.static_objects.confidence
        _bboxes: Optional[List[List[int]]] = self.static_objects.bbox
        match_labels: Optional[List[str]] = g.config.matching.static_objects.labels
        mon_filt = g.config.monitors.get(g.mid)
        zone_filt: Optional[MonitorZones] = None
        if mon_filt and zone_name in mon_filt.zones:
            zone_filt = mon_filt.zones[zone_name]

        # Override with monitor filters than zone filters
        if mon_filt and mon_filt.static_objects.difference:
            _max_diff_area = mon_filt.static_objects.difference
        if zone_filt and zone_filt.static_objects.difference:
            _max_diff_area = zone_filt.static_objects.difference
        if mon_filt and mon_filt.static_objects.labels:
            match_labels = mon_filt.static_objects.labels
        if zone_filt and zone_filt.static_objects.labels:
            match_labels = zone_filt.static_objects.labels

        ignore_labels: Optional[List[str]] = (
            g.config.matching.static_objects.ignore_labels or []
        )

        # Do zones first as they take precedence over monitor filters
        if zone_filt and zone_filt.static_objects.ignore_labels:
            for lbl in zone_filt.static_objects.ignore_labels:
                if lbl not in ignore_labels:
                    ignore_labels.append(lbl)
        elif mon_filt and mon_filt.static_objects.ignore_labels:
            for lbl in mon_filt.static_objects.ignore_labels:
                if lbl not in ignore_labels:
                    ignore_labels.append(lbl)

        if ignore_labels and current_label in ignore_labels:
            logger.debug(
                f"{lp} {current_label} is in static_objects:ignore_labels: {ignore_labels}, skipping",
            )
        else:
            if match_labels is None:
                logger.debug(
                    f"{lp} no labels configured in static_objects:labels:, default behavior is to check all labels",
                )
                match_labels = [current_label]
            if current_label in match_labels:
                logger.debug(
                    f"{lp} max difference between current and past object area configured -> {_max_diff_area}"
                )
                if isinstance(_max_diff_area, float):
                    if _max_diff_area >= 1.0:
                        _max_diff_area = 1.0
                elif isinstance(_max_diff_area, int):
                    pass

                else:
                    logger.warning(
                        f"{lp} Unknown type for difference, defaulting to 5%"
                    )
                    _max_diff_area = 0.05
                if _labels:
                    for saved_label, saved_conf, saved_bbox in zip(
                        _labels, _confs, _bboxes
                    ):
                        # compare current detection element with saved list from file
                        found_alias_grouping = False
                        # check if it is in a label group
                        if saved_label != current_label:
                            # 'label group' support
                            if aliases:
                                logger.debug(
                                    f"{lp} currently detected object does not match saved object, "
                                    f"checking label_groups for an aliased match"
                                )

                                for alias, alias_group in aliases.items():
                                    if (
                                        saved_label in alias_group
                                        and current_label in alias_group
                                    ):
                                        logger.debug(
                                            f"{lp} saved and current object are in the same label group [{alias}]"
                                        )
                                        found_alias_grouping = True
                                        break

                        elif saved_label == current_label:
                            found_alias_grouping = True
                        if not found_alias_grouping:
                            logger.debug(
                                f"{lp} saved and current object are not equal or in the same label group, skipping"
                            )
                            continue
                        # Found a match by label/group, now compare the area using Polygon
                        try:
                            past_label_polygon = Polygon(self._bbox2points(saved_bbox))
                        except Exception as e:
                            logger.error(
                                f"{lp} Error converting saved_bbox to polygon: {e}, skipping"
                            )
                            continue
                        max_diff_pixels = None
                        diff_area = None
                        logger.debug(
                            f"{lp} comparing '{current_label}' PAST->{past_label_polygon} to CURR->{list(zip(*current_bbox_polygon.exterior.coords.xy))[:-1]}",
                        )
                        if past_label_polygon.intersects(
                            current_bbox_polygon
                        ) or current_bbox_polygon.intersects(past_label_polygon):
                            if past_label_polygon.intersects(current_bbox_polygon):
                                logger.debug(
                                    f"{lp} the PAST object INTERSECTS the new object",
                                )
                            else:
                                logger.debug(
                                    f"{lp} the current object INTERSECTS the PAST object",
                                )

                            if current_bbox_polygon.contains(past_label_polygon):
                                diff_area = current_bbox_polygon.difference(
                                    past_label_polygon
                                ).area
                                if isinstance(_max_diff_area, float):
                                    max_diff_pixels = (
                                        current_bbox_polygon.area * _max_diff_area
                                    )
                                    logger.debug(
                                        f"{lp} converted {_max_diff_area * 100:.2f}% difference from '{current_label}' "
                                        f"is {max_diff_pixels} pixels"
                                    )
                                elif isinstance(_max_diff_area, int):
                                    max_diff_pixels = _max_diff_area
                            else:
                                diff_area = past_label_polygon.difference(
                                    current_bbox_polygon
                                ).area
                                if isinstance(_max_diff_area, float):
                                    max_diff_pixels = (
                                        past_label_polygon.area * _max_diff_area
                                    )
                                    logger.debug(
                                        f"{lp} converted {_max_diff_area * 100:.2f}% difference from '{saved_label}' "
                                        f"is {max_diff_pixels} pixels"
                                    )
                                elif isinstance(_max_diff_area, int):
                                    max_diff_pixels = _max_diff_area
                            if diff_area is not None and diff_area <= max_diff_pixels:
                                # FIXME: if an object is removed, the PAST object it matched should be propagated to the next event
                                logger.debug(
                                    f"{lp} removing '{current_label}' as it seems to be approximately in the same spot"
                                    f" as it was detected last time based on '{_max_diff_area}' -> Difference in pixels: {diff_area} "
                                    f"- Configured maximum difference in pixels: {max_diff_pixels}"
                                )
                                _past_match = (saved_label, saved_conf, saved_bbox)
                                propagations = g.static_objects.get("propagate")
                                if propagations is None:
                                    propagations = []
                                if propagations:
                                    if _past_match not in propagations:
                                        propagations.append(_past_match)

                                return False

                                # if saved_bbox not in mpd_b:
                                #     logger.debug(
                                #         f"{lp} appending this saved object to the mpd "
                                #         f"buffer as it has removed a detection and should be propagated "
                                #         f"to the next event"
                                #     )
                                #     mpd_b.append(saved_bs[saved_idx])
                                #     mpd_l.append(saved_label)
                                #     mpd_c.append(saved_cs[saved_idx])
                                # new_err.append(b)
                            elif diff_area is not None and diff_area > max_diff_pixels:
                                logger.debug(
                                    f"{lp} allowing '{current_label}' -> the difference in the area of last detection "
                                    f"to this detection is '{diff_area:.2f}', a minimum of {max_diff_pixels:.2f} "
                                    f"is needed to not be considered 'in the same spot'",
                                )
                                return True
                            elif diff_area is None:
                                logger.debug(
                                    f"DEBUG>>>'MPD' {diff_area = } - whats the issue?"
                                )
                            else:
                                logger.debug(
                                    f"WHATS GOING ON? {diff_area = } -- {max_diff_pixels = }"
                                )
                        # Saved does not intersect the current object/label
                        else:
                            logger.debug(
                                f"{lp} current detection '{current_label}' is not near enough to '"
                                f"{saved_label}' to evaluate for match past detection filter"
                            )
                else:
                    logger.debug(
                        f"{lp} no saved detections to compare to, allowing '{current_label}'"
                    )
            else:
                logger.debug(
                    f"{lp} '{current_label}' is not in static_objects:labels: {match_labels}, skipping..."
                )
        return True

    @staticmethod
    def construct_filter(filters: Dict) -> OverRideMatchFilters:
        # construct each object label filter
        if filters["object"]["labels"]:
            for label_ in filters["object"]["labels"]:
                filters["object"]["labels"][
                    label_
                ] = OverRideObjectFilters.model_construct(
                    **filters["object"]["labels"][label_],
                )

        filters["object"] = OverRideObjectFilters.model_construct(
            **filters["object"],
        )
        filters["face"] = OverRideFaceFilters.model_construct(
            **filters["face"],
        )
        filters["alpr"] = OverRideAlprFilters.model_construct(
            **filters["alpr"],
        )
        return OverRideMatchFilters.model_construct(**filters)

    def load_config(self) -> Optional[ConfigFileModel]:
        """Parse the YAML configuration file. In the future this will read DB values"""
        cfg: Dict = {}
        _start = time.time()
        self.raw_config = self.config_file.read_text()

        try:
            cfg = yaml.safe_load(self.raw_config)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing the YAML configuration file!")
            raise e

        substitutions = cfg.get("substitutions", {})
        testing = cfg.get("testing", {})
        testing = Testing(**testing)
        if testing.enabled:
            logger.info(f"|----- TESTING IS ENABLED! -----|")
            if testing.substitutions:
                logger.info(
                    f"Overriding config:substitutions WITH testing:substitutions"
                )
                substitutions = testing.substitutions

        logger.debug(f"Replacing ${{VARS}} in config:substitutions")
        substitutions = self.replace_vars(str(substitutions), substitutions)
        if inc_file := substitutions.get("IncludeFile"):
            inc_file = Path(inc_file)
            logger.debug(f"PARSING IncludeFile: {inc_file.as_posix()}")
            if inc_file.is_file():
                inc_vars = yaml.safe_load(inc_file.read_text())
                if "client" in inc_vars:
                    inc_vars = inc_vars.get("client", {})
                    logger.debug(
                        f"Loaded {len(inc_vars)} substitution from IncludeFile {inc_file} => {inc_vars}"
                    )
                    # check for duplicates
                    for k in inc_vars:
                        if k in substitutions:
                            logger.warning(
                                f"Duplicate substitution variable '{k}' in IncludeFile {inc_file} - "
                                f"IncludeFile overrides config file"
                            )

                    substitutions.update(inc_vars)
                else:
                    logger.warning(
                        f"IncludeFile [{inc_file}] does not have a 'client' section - skipping"
                    )
            else:
                logger.warning(f"IncludeFile {inc_file} is not a file!")
        logger.debug(f"Replacing ${{VARS}} in config")
        cfg = self.replace_vars(self.raw_config, substitutions)
        self.parsed_cfg = dict(cfg)
        _x = ConfigFileModel(**cfg)
        logger.debug(
            f"perf:: Config file loaded and validated in {time.time() - _start:.5f} seconds"
        )
        return _x

    @staticmethod
    def replace_vars(search_str: str, var_pool: Dict) -> Dict:
        """Replace variables in a string.


        Args:
            search_str (str): String to search for variables '${VAR_NAME}'.
            var_pool (Dict): Dictionary of variables used to replace.

        """
        import re

        if var_list := re.findall(r"\$\{(\w+)\}", search_str):
            # $ remove duplicates
            var_list = list(set(var_list))
            logger.debug(f"Found the following substitution variables: {var_list}")
            # substitute variables
            _known_vars = []
            _unknown_vars = []
            for var in var_list:
                if var in var_pool:
                    # logger.debug(
                    #     f"substitution variable '{var}' IS IN THE POOL! VALUE: "
                    #     f"{var_pool[var]} [{type(var_pool[var])}]"
                    # )
                    _known_vars.append(var)
                    value = var_pool[var]
                    if value is None:
                        value = ""
                    elif value is True:
                        value = "yes"
                    elif value is False:
                        value = "no"
                    search_str = search_str.replace(f"${{{var}}}", value)
                else:
                    _unknown_vars.append(var)
            if _unknown_vars:
                logger.warning(
                    f"The following variables have no configured substitution value: {_unknown_vars}"
                )
            if _known_vars:
                logger.debug(
                    f"The following variables have been substituted: {_known_vars}"
                )
        else:
            logger.debug(f"No substitution variables found.")

        return yaml.safe_load(search_str)

    def send_notifications(
        self, noti_img: np.ndarray, prediction_str: str, results: Optional = None
    ):
        lp = f"notifications::"
        noti_cfg = g.config.notifications
        if any(
            [
                noti_cfg.gotify.enabled,
                noti_cfg.zmninja.enabled,
                noti_cfg.mqtt.enabled,
                noti_cfg.pushover.enabled,
                noti_cfg.shell_script.enabled,
            ]
        ):
            futures: List[concurrent.futures.Future] = []
            with concurrent.futures.ThreadPoolExecutor(
                thread_name_prefix="notifications",
                max_workers=g.config.system.thread_workers,
            ) as executor:
                if noti_cfg.pushover.enabled:
                    # Pushover has a limit of messages per month, so it needs a one time strategy
                    # Pushover requires to send the image/gif to them instead of requesting it from the server
                    po = self.notifications.pushover
                    _cfg = noti_cfg.pushover
                    po.request_data.token = _cfg.token
                    po.request_data.user = _cfg.key
                    po.request_data.message = f"{prediction_str.strip()}"
                    po.request_data.title = f"({g.eid}) {g.mon_name}->{g.event_cause}"
                    po.request_data.priority = _cfg.priority
                    po.request_data.html = 1
                    po.request_data.timestamp = time.time()
                    if noti_cfg.pushover.clickable_link:
                        po.request_data.url_title = "View event in browser"
                        push_url_opts: NotificationZMURLOptions = (
                            noti_cfg.pushover.url_opts
                        )
                        _mode = push_url_opts.mode
                        _scale = push_url_opts.scale
                        _max_fps = push_url_opts.max_fps
                        _buffer = push_url_opts.buffer
                        _replay = push_url_opts.replay

                        view_url = (
                            f"{g.api.portal_base_url}/cgi-bin/nph-zms?mode={_mode}&scale="
                            f"{_scale}&maxfps={_max_fps}&buffer={_buffer}&replay={_replay}&"
                            f"monitor={g.mid}&event={g.eid}"
                        )
                        if _auth := po._push_auth:
                            view_url = f"{view_url}&{_auth}"
                        po.request_data.url = view_url

                    # try:
                    #     # do custom sound
                    #     po.parse_sounds(matches["labels"])
                    # except Exception as exc:
                    #     logger.error(f"{lp} failed to parse sounds: {exc}")
                    # else:
                    #     logger.debug(f"PROPERLY parsed sounds for JPEG pushover?")
                    #     display_param_dict["sounds"] = po.request_data.sound

                    # swap RB channels for pushover
                    po.image = cv2.cvtColor(noti_img, cv2.COLOR_BGR2RGB)
                    po.optionals.cache_write = False if g.past_event else True
                    futures.append(executor.submit(po.send))
                    logger.debug(f"{lp} Pushover notification configured")

                if noti_cfg.gotify.enabled:
                    logger.debug(f"{lp} Gotify notification configured")
                    goti = self.notifications.gotify
                    # gotify has no limits, so it can send a notification for each frame
                    goti.title = f"({g.eid}) {g.mon_name}->{g.event_cause}"
                    # goti.send(prediction_str)
                    futures.append(executor.submit(goti.send, prediction_str))

                if noti_cfg.zmninja.enabled:
                    # zmninja uses FCM which has a limit of messages per month, so it needs a one time strategy
                    logger.debug(f"{lp} ZMNinja notification configured, sending")
                    # self.notifications.zmninja.send()

                if noti_cfg.mqtt.enabled:
                    logger.debug(f"{lp} MQTT notification configured")
                    mqtt_results = {
                        "labels": results["labels"],
                        "model_names": results["model_names"],
                        "confidences": results["confidences"],
                        "frame_id": results["frame_id"],
                        "detection_types": results["detection_types"],
                        "bounding_boxes": results["bounding_boxes"],
                        "processor": results["processor"],
                        "mid": g.mid,
                        "eid": g.eid,
                    }
                    # self.notifications.mqtt.publish(
                    #     fmt_str=prediction_str,
                    #     results=mqtt_results,
                    #     image=noti_img,
                    #     mid=g.mid,
                    # )
                    futures.append(
                        executor.submit(
                            self.notifications.mqtt.publish,
                            fmt_str=prediction_str,
                            results=mqtt_results,
                            image=noti_img,
                            mid=g.mid,
                        )
                    )

                if noti_cfg.shell_script.enabled:
                    logger.debug(f"{lp} Shell Script notification configured, sending")
                    futures.append(
                        executor.submit(
                            self.notifications.shell_script.send,
                            prediction_str,
                            results,
                        )
                    )
                    # try:
                    #     self.notifications.shell_script.send(prediction_str, results)
                    # except Exception as exc:
                    #     logger.error(f"{lp} failed to send notification: {exc}")
                    # else:
                    #     logger.debug(f"{lp} Shell Script notification completed")

            for future in concurrent.futures.as_completed(futures):
                try:
                    exc_ = future.exception(timeout=10)
                    if exc_:
                        raise exc_
                except Exception as exc:
                    logger.error(f"{lp} failed to send notification: {exc}", exc_info=True)
                    # raise exc
                else:
                    future.result()
        else:
            logger.debug(f"{lp} No notifications configured, skipping")
        del noti_img

    async def post_process(self, matches: Dict[str, Any]) -> None:
        perf_postproc_start = time.time()
        labels, scores, boxes = (
            matches["labels"],
            matches["confidences"],
            matches["bounding_boxes"],
        )
        # check if bounding boxes contains any str
        _skip = False
        if any(isinstance(x, str) for x in boxes):
            # there is a bbox that is a str instead of a tuple
            _skip = True

        model, processor = matches["model_names"], matches["processor"]
        image: np.ndarray = matches["frame_img"]
        prepared_image = image.copy()
        image_name = str(matches["frame_id"])
        # annotate the image
        lp = f"post process::"
        from .Models.utils import draw_bounding_boxes

        write_conf = g.config.detection_settings.images.annotation.confidence
        write_model = g.config.detection_settings.images.annotation.model.enabled
        write_processor = g.config.detection_settings.images.annotation.model.processor

        if _skip:
            logger.debug(f"{LP} No need to annotate, grabbing pre annotated image")
        else:
            if g.config.detection_settings.images.annotation.zones.enabled:
                from .Models.utils import draw_zones

                logger.debug(f"{lp} Annotating zones")
                prepared_image = draw_zones(
                    prepared_image,
                    self.zone_polygons,
                    g.config.detection_settings.images.annotation.zones.color,
                    g.config.detection_settings.images.annotation.zones.thickness,
                    g.config.detection_settings.images.annotation.zones.show_name,
                )
            logger.debug(f"{lp} Annotating detections")
            prepared_image: np.ndarray = draw_bounding_boxes(
                prepared_image,
                labels=labels,
                confidences=scores,
                boxes=boxes,
                model=model,
                processor=processor,
                write_conf=write_conf,
                write_model=write_model,
                write_processor=write_processor,
            )
        if g.config.detection_settings.images.debug.enabled:
            from .Models.utils import draw_filtered_bboxes

            logger.debug(f"{lp} Debug image configured, drawing filtered out bboxes")

            debug_image = draw_filtered_bboxes(
                prepared_image, list(self.filtered_labels[image_name])
            )
            logger.debug(f"DBG:FIX ME>>> {list(self.filtered_labels[image_name])}")
            from datetime import datetime

            if g.config.detection_settings.images.debug.path:
                _dest = g.config.detection_settings.images.debug.path
                logger.debug(f"{lp} Debug image PATH configured: {_dest.as_posix()}")
            elif g.config.system.image_dir:
                _dest = g.config.system.image_dir
                logger.debug(
                    f"{lp} Debug image path NOT configured, using system image_dir: {_dest.as_posix()}"
                )
            else:
                _dest = g.config.system.variable_data_path / "images"
                logger.debug(
                    f"{lp} Debug image path and system image_dir NOT configured"
                    f" using {{system:variable_data_dir}} as base: {_dest.as_posix()}"
                )

            img_write_success = cv2.imwrite(
                _dest.joinpath(f"debug-img_{datetime.now()}.jpg").as_posix(),
                debug_image,
            )
            if img_write_success:
                logger.debug(f"{lp} Debug image written to disk.")
            else:
                logger.warning(f"{lp} Debug image failed to write to disk.")
            del debug_image

        jpg_file = g.event_path / "objdetect.jpg"
        object_file = g.event_path / "objects.json"
        try:
            objdetect_jpg = cv2.imwrite(jpg_file.as_posix(), prepared_image)
        except Exception as write_img_exc:
            logger.error(
                f"{lp} objdetect.jpg failed to write to disk: err_msg=> \n{write_img_exc}\n"
            )
        else:
            if objdetect_jpg:
                logger.debug(f"{lp} objdetect.jpg written to disk @ '{jpg_file}'")
            else:
                logger.warning(f"{lp} objdetect.jpg failed to write to disk.")

        obj_json = {
            "frame_id": image_name,
            "labels": labels,
            "confidences": scores,
            "boxes": boxes,
            "image_dimensions": image.shape,
        }
        try:
            json.dump(obj_json, object_file.open("w"))
        except Exception as custom_push_exc:
            logger.error(
                f"{lp} objects.json failed to write to disk: err_msg=> \n{custom_push_exc}\n"
            )
        else:
            logger.debug(f"{lp} objects.json written to disk @ '{object_file}'")

        _frame_id = matches["frame_id"]
        prefix = f"[{_frame_id}] "
        model_names: list = matches["model_names"]
        # Construct the prediction text
        seen = []
        pred = ""
        # add the label, confidence and model name if configured
        for _l, _c, _b in zip(labels, scores, boxes):
            if _l not in seen:
                label_txt = f"{_l}"
                model_txt = (
                    model_names
                    if g.config.detection_settings.images.annotation.model.enabled
                    else ""
                )
                processor_txt = (
                    f"[{processor}]"
                    if g.config.detection_settings.images.annotation.model.processor
                    and processor
                    else ""
                )
                conf_txt = (
                    f" ({_c:.2f})"
                    if g.config.detection_settings.images.annotation.confidence
                    else ""
                )
                if not g.config.detection_settings.images.annotation.confidence:
                    label_txt = f"{_l}{f'({model_txt})' if model_txt else ''}{processor_txt if processor_txt else ''}, "
                else:
                    label_txt = (
                        f"{_l}({_c:.0%}"
                        f"{f'-{model_txt}' if model_txt else ''}{processor_txt if processor_txt else ''}), "
                    )
                pred = f"{pred}{label_txt}"
                seen.append(_l)

        # :detected: needs to be in the Notes field for the Event in ZM
        pred = pred.strip().rstrip(",")  # remove trailing comma
        pred_out = f":detected:{pred}"
        pred = f"{pred}"
        new_notes = pred_out
        logger.info(f"{lp}prediction: '{pred}'")
        # Check notes and replace if necessary
        old_notes: str = g.db.event_notes(g.eid)
        new_notes = f"{new_notes} {g.event_cause}"
        _write_notes = False
        if old_notes is not None:
            if new_notes != old_notes:
                _write_notes = True
            elif new_notes == old_notes:
                logger.debug(f"{lp} notes do not need updating!")
        else:
            _write_notes = True

        if _write_notes:
            try:
                g.db.set_notes(g.eid, new_notes)
            except Exception as custom_push_exc:
                logger.error(
                    f"{lp} error during notes DB update request-> {custom_push_exc}"
                )
            else:
                logger.debug(
                    f"{lp} replaced old note with: '{new_notes}'",
                )

        # Check version of zm, if 1.37.44 or greater, we can manage tags
        zm_ver = g.db.get_zm_version()
        tags_support = False
        if zm_ver.major == 1:
            if zm_ver.minor == 37:
                if zm_ver.patch >= 44:
                    tags_support = True
            elif zm_ver.minor > 37:
                tags_support = True
        elif zm_ver.major > 1:
            tags_support = True

        # if tags_support:
        #     # check that the detected object label has a tag created
        #
        #     # check if the event has tags
        #     tags = g.db.get_event_tags(g.eid)
        #     if tags:
        #         # check if the event has the detected labels tag
        #         if "detected" not in tags:
        #             # add the detected tag
        #             tags.append("detected")
        #             g.db.set_event_tags(g.eid, tags)
        #     else:
        #         # add the detected tag
        #         tags = ["detected"]
        #         g.db.set_event_tags(g.eid, tags)

        # send notifications
        self.send_notifications(prepared_image, pred_out, results=matches)
