import logging
import re
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    List,
    Tuple,
    Pattern,
    Union,
    Any,
    Optional,
    AnyStr, Annotated,
)

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    AnyUrl,
    field_validator,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .Enums import ModelProcessor, ModelType
from .validators import (
    validate_percentage_or_pixels,
    validate_resolution,
    validate_points,
    validate_no_scheme_url,
    validate_file,
    validate_dir,
    validate_log_level,
    str2path,
    validate_enabled,
    validate_not_enabled,
)
from ..Libs.API import ZMAPI
from ..Libs.DB import ZMDB
from ..Log import CLIENT_LOGGER_NAME
from ..Models.DEFAULTS import *

logger = logging.getLogger(CLIENT_LOGGER_NAME)


class Testing(BaseModel):
    enabled: bool = Field(False)
    substitutions: Dict[str, str] = Field(default_factory=dict)


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)

    _v = field_validator("enabled", mode="before")(validate_enabled)


class DefaultNotEnabled(BaseModel):
    enabled: bool = Field(False)

    _v = field_validator("enabled", mode="before")(validate_not_enabled)


class LoggingLevelBase(BaseModel):
    level: Optional[int] = None

    _validate_log_level = field_validator("level", mode="before")(validate_log_level)


class LoggingSettings(LoggingLevelBase):
    class ConsoleLogging(DefaultEnabled, LoggingLevelBase):
        pass

    class SyslogLogging(DefaultNotEnabled, LoggingLevelBase):
        address: Optional[str] = Field("")

    class FileLogging(DefaultEnabled, LoggingLevelBase):
        path: Path = Path(DEF_CLNT_LOGGING_FILE_PATH)
        filename_prefix: str = DEF_CLNT_LOGGING_FILE_FILENAME_PREFIX
        file_name: Optional[str] = None
        user: Optional[str] = Field(None)
        group: Optional[str] = Field(None)

        _validate_path = field_validator("path", mode="before")(str2path)

    class SanitizeLogging(DefaultNotEnabled):
        replacement_str: str = Field(default="<sanitized>")

    level: int = logging.INFO
    console: ConsoleLogging = Field(default_factory=ConsoleLogging)
    syslog: SyslogLogging = Field(default_factory=SyslogLogging)
    file: FileLogging = Field(default_factory=FileLogging)
    sanitize: SanitizeLogging = Field(default_factory=SanitizeLogging)


class ZMDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_CLIENT_DB_", extra="allow")
    host: Union[None, AnyStr] = Field(None)
    port: Optional[int] = Field(None)
    user: Optional[str] = Field(None)
    password: Optional[SecretStr] = Field(None)
    name: Optional[str] = Field(None)
    driver: Optional[str] = Field(None)


class SystemSettings(BaseModel):
    image_dir: Optional[Path] = Field(Path(DEF_CLNT_SYS_IMAGEDIR))
    config_path: Optional[Path] = Field(Path(DEF_CLNT_SYS_CONFDIR))
    variable_data_path: Optional[Path] = Field(DEF_CLNT_SYS_DATADIR)
    tmp_path: Optional[Path] = Field(Path(DEF_CLNT_SYS_TMPDIR))
    venv_path: Optional[Path] = None
    thread_workers: Optional[int] = Field(DEF_CLNT_SYS_THREAD_WORKERS)


class ZMSPullMethod(DefaultNotEnabled):
    sbf: Optional[int] = Field(
        None,
        description="Seconds between frame grabs for LIVE events"
        " (1 would = 1 fps, 2 = .5 fps).",
    )
    url: Optional[AnyUrl] = Field(
        None,
        description="URL to the nph-zms cgi script (ex: http://zm.example.com/zm/cgi-bin/nph-zms). If not supplied it will be auto-detected.",
    )
    attempts: Optional[int] = Field(3, description="Number of attempts to get a frame from a PAST event")
    delay: Optional[float] = Field(1.0, description="Delay between attempts on a PAST event")
    max_frames: Optional[int] = Field(
        0, description="Maximum number of frames to process (LIVE/PAST)"
    )
    timeout: Optional[int] = Field(10)

    lp_: str = Field("", description="logging prefix", repr=False)

    @model_validator(mode="after")
    def _validate_model_after(self):
        self.lp_ = f"{self.__class__.__name__}:"
        return self


class APIPullMethod(DefaultNotEnabled):
    fps: Optional[int] = Field(
        1, description="Frames per second to capture. Cannot be used with sbf."
    )
    sbf: Optional[int] = Field(
        None,
        description="Seconds between frame "
        "(for image sources that only update every <x> seconds). "
        "Cannot be used with fps.",
    )
    attempts: Optional[int] = Field(3)
    delay: Optional[float] = Field(1.0)
    check_snapshots: Optional[bool] = Field(True)
    snapshot_frame_skip: Optional[int] = Field(3)
    max_frames: Optional[int] = Field(0)
    timeout: Optional[int] = Field(10)
    lp_: str = Field("", description="logging prefix", repr=False)

    @model_validator(mode="after")
    def _validate_model_after(self):
        self.lp_ = f"{self.__class__.__name__}:"
        if self.fps and self.sbf:
            logger.warning(
                f"{self.lp_} fps and sbf cannot both be set, sbf takes precedence"
            )
            self.fps = None
        elif not self.fps and not self.sbf:
            logger.warning(
                f"{self.lp_} fps and sbf are both not set, defaulting to 1 fps"
            )
            self.fps = 1
        return self


class PullMethod(BaseModel):
    api: Optional[APIPullMethod] = None
    zms: Optional[ZMSPullMethod] = None


class PullMethod(BaseModel):
    api: Optional[APIPullMethod] = Field(default_factory=APIPullMethod)
    zms: Optional[ZMSPullMethod] = Field(default_factory=ZMSPullMethod)

    _validate_ = field_validator("zms", "api", mode="before")(
        validate_not_enabled
    )


class ZoneMinderSettings(BaseSettings, extra="allow"):
    model_config = SettingsConfigDict(env_prefix="ML_CLIENT_ZM_")

    class ZMMisc(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="ML_CLIENT_ZM_MISC_")
        write_notes: bool = Field(True)

    class ZMAPISettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="ML_CLIENT_ZM_API_", extra="allow"
        )
        api_url: Optional[AnyUrl] = Field(None)
        user: Optional[SecretStr] = Field(None)
        password: Optional[SecretStr] = Field(None)
        ssl_verify: Optional[bool] = Field(True)
        headers: Optional[Dict] = Field(default_factory=dict)

        _validate_api_url = field_validator("api_url", mode="before")(
            validate_no_scheme_url
        )

    conf_dir: Path = Field(
        Path("/etc/zm"),
        description="Path to ZoneMinder config files, Default: /etc/zm",
    )
    portal_url: Optional[AnyUrl] = Field(None)
    misc: Optional[ZMMisc] = Field(default_factory=ZMMisc)
    api: Optional[ZMAPISettings] = Field(default_factory=ZMAPISettings)
    db: Optional[ZMDBSettings] = Field(default_factory=ZMDBSettings)
    pull_method: Optional[PullMethod] = Field(default_factory=ZMSPullMethod)


    _validate_portal_url = field_validator("portal_url", mode="before")(
        validate_no_scheme_url
    )


class ServerRoute(BaseModel):
    # Make 1 attr required so empty entries will fail.
    name: str = Field(...)
    host: AnyUrl = Field(...)
    port: Optional[Union[int, str]] = Field(5000)
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    timeout: Optional[int] = Field(90, ge=0)

    _validate_mlapi_host_no_scheme = field_validator("host", mode="before")(
        validate_no_scheme_url
    )

    @model_validator(mode="after")
    def _validate_model_after(self):
        if not self.port:
            self.port = 5000
        return self


class AnimationSettings(BaseModel):
    gif: bool = Field(False)
    mp4: bool = Field(False)
    width: Optional[int] = Field(640)
    fast_gif: bool = Field(False)
    low_memory: Optional[bool] = Field(False)
    overwrite: Optional[bool] = Field(False)
    max_attempts: Optional[int] = Field(ge=1, default=3)
    attempt_delay: Optional[float] = Field(
        ge=0.1, default=2.9, description="Delay between attempts in seconds"
    )


class NotificationZMURLOptions(BaseModel):
    mode: Optional[str] = Field("jpeg")
    scale: Optional[int] = Field(50)
    max_fps: Optional[int] = Field(15)
    buffer: Optional[int] = Field(1000)
    replay: Optional[str] = Field("single")


class CoolDownSettings(DefaultNotEnabled):
    seconds: Optional[float] = Field(
        60.00, ge=0.0, description="Seconds to wait before sending another notification"
    )


class OverRideCoolDownSettings(CoolDownSettings):
    linked: Optional[list[str]] = Field(
        default_factory=list, description="List of linked monitors"
    )


class MLNotificationSettings(BaseModel):
    class ZMNinjaNotificationSettings(DefaultEnabled):
        class ZMNinjaFCMSettings(BaseModel):
            class FCMV1Settings(DefaultEnabled):
                key: Optional[SecretStr] = None
                url: Optional[AnyUrl] = None

            v1: Optional[FCMV1Settings] = Field(default_factory=FCMV1Settings)
            token_file: Optional[Path] = None
            replace_messages: Optional[bool] = Field(False)
            date_fmt: Optional[str] = Field("%I:%M %p, %d-%b")
            android_priority: Optional[str] = Field("high")
            log_raw_message: Optional[bool] = Field(False)
            log_message_id: Optional[str] = None
            android_ttl: Optional[int] = Field(0)

        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)
        fcm: Optional[ZMNinjaFCMSettings] = Field(default_factory=ZMNinjaFCMSettings)

    class GotifyNotificationSettings(DefaultNotEnabled):
        test_image: Optional[bool] = Field(False)
        host: Optional[AnyUrl] = None
        token: Optional[str] = None
        portal: Optional[AnyUrl] = None
        clickable_link: Optional[bool] = Field(False)
        link_user: Optional[SecretStr] = None
        link_pass: Optional[SecretStr] = None
        _push_auth: Optional[SecretStr] = None
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)

        url_opts: Optional[NotificationZMURLOptions] = Field(
            default_factory=NotificationZMURLOptions
        )

        # validators
        _validate_host_portal = field_validator("host", "portal", mode="before")(
            validate_no_scheme_url
        )

    class PushoverNotificationSettings(BaseModel):
        class SendAnimations(DefaultNotEnabled):
            token: Optional[str] = None
            key: Optional[str] = None

        class EndPoints(BaseModel):
            messages: Optional[str] = Field("/messages.json")
            users: Optional[str] = Field("/users/validate.json")
            devices: Optional[str] = Field("/devices.json")
            sounds: Optional[str] = Field("/sounds.json")
            receipt: Optional[str] = Field("/receipts/{receipt}.json")
            cancel: Optional[str] = Field("/cancel/{receipt}.json")
            emergency: Optional[str] = Field("/emergency.json")

        enabled: Optional[bool] = Field(False)
        token: Optional[str] = None
        key: Optional[str] = None
        animation: Optional[SendAnimations] = Field(default_factory=SendAnimations)
        sounds: Optional[Dict[str, str]] = Field(default_factory=dict)
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)
        device: Optional[str] = None
        url_opts: Optional[NotificationZMURLOptions] = Field(
            default_factory=NotificationZMURLOptions
        )
        base_url: Optional[AnyUrl] = Field("https://api.pushover.net/1")
        endpoints: Optional[EndPoints] = Field(default_factory=EndPoints)
        clickable_link: Optional[bool] = Field(False)
        link_user: Optional[SecretStr] = None
        link_pass: Optional[SecretStr] = None
        priority: Optional[int] = Field(ge=-2, le=2, default=0)

        # validators
        _validate_host_portal = field_validator("base_url", mode="before")(
            validate_no_scheme_url
        )

    class ShellScriptNotificationSettings(DefaultNotEnabled):
        script: Optional[str] = None
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)
        I_AM_AWARE_OF_THE_DANGER_OF_RUNNING_SHELL_SCRIPTS: Optional[str] = "No I am not"
        # TODO: ARGS and AUTH
        pass_token: Optional[bool] = Field(False, description="Pass JWT to script")
        pass_creds: Optional[bool] = Field(
            False, description="Pass username and password to script"
        )
        args: Optional[List[str]] = None

    class MQTTNotificationSettings(DefaultNotEnabled):
        class MQTTImageSettings(DefaultNotEnabled):
            format: Optional[str] = Field("base64", pattern="^(bytes|base64)$")
            retain: Optional[bool] = True

        keep_alive: Optional[int] = Field(60, ge=1)
        root_topic: Optional[str] = Field("zomi-client")
        broker: Optional[str] = None
        port: Optional[int] = Field(1883)
        user: Optional[str] = None
        pass_: Optional[SecretStr] = Field(None, alias="pass")
        tls_secure: Optional[bool] = Field(True)
        tls_ca: Optional[Path] = None
        tls_cert: Optional[Path] = None
        tls_key: Optional[Path] = None
        retain: Optional[bool] = Field(False)
        qos: Optional[int] = Field(0)
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)

        image: Optional[MQTTImageSettings] = Field(default_factory=MQTTImageSettings)

    mqtt: Optional[MQTTNotificationSettings] = Field(
        default_factory=MQTTNotificationSettings
    )
    zmninja: Optional[ZMNinjaNotificationSettings] = Field(
        default_factory=ZMNinjaNotificationSettings
    )
    gotify: Optional[GotifyNotificationSettings] = Field(
        default_factory=GotifyNotificationSettings
    )
    pushover: Optional[PushoverNotificationSettings] = Field(
        default_factory=PushoverNotificationSettings
    )
    shell_script: Optional[ShellScriptNotificationSettings] = Field(
        default_factory=ShellScriptNotificationSettings
    )


class ColorDetectionSettings(DefaultNotEnabled):
    top_n: Optional[int] = Field(3)
    labels: Optional[List[Optional[str]]] = None


class DetectionSettings(BaseModel):
    class ImageSettings(BaseModel):
        class Debug(DefaultNotEnabled):
            path: Optional[Path] = Field(Path("/tmp"))

        class Annotations(BaseModel):
            class Zones(DefaultNotEnabled):
                color: Union[Tuple[int, int, int], None] = Field(
                    (255, 0, 0), description="Color of polygon line in BGR"
                )
                thickness: Optional[int] = Field(2)
                show_name: Optional[bool] = Field(
                    False, description="Overlay the zone name on the image"
                )

                @field_validator("color", mode="before")
                def _validate_color(cls, v):
                    if v:
                        if isinstance(v, str):
                            v = tuple(
                                int(x) for x in v.lstrip("(").rstrip(")").split(",")
                            )
                        if not isinstance(v, tuple) or len(v) != 3:
                            raise ValueError("Must be a tuple of 3 integers")
                        for x in v:
                            if not isinstance(x, int) or x < 0 or x > 255:
                                raise ValueError(
                                    "Must be a tuple of 3 integers between 0 and 255"
                                )
                    assert isinstance(v, (tuple, None))
                    return v

            class Models(DefaultEnabled):
                processor: Optional[bool] = Field(False)

            zones: Optional[Zones] = Field(default_factory=Zones)
            model: Optional[Models] = Field(default_factory=Models)

            confidence: Optional[bool] = Field(True)

        class Training(DefaultEnabled):
            enabled: Optional[bool] = Field(False)
            path: Optional[Path] = Field(DEF_CLNT_SYS_IMAGEDIR + "train/")

        pull_method: Optional[PullMethod] = Field(default_factory=PullMethod)
        debug: Optional[Debug] = Field(default_factory=Debug)
        annotation: Optional[Annotations] = Field(default_factory=Annotations)
        training: Optional[Training] = Field(default_factory=Training)

    models: Optional[Dict] = Field(default_factory=dict)
    motion_only: Optional[bool] = Field(True)
    import_zones: Optional[bool] = Field(False)
    match_origin_zone: Optional[bool] = Field(False)
    images: Optional[ImageSettings] = Field(default_factory=ImageSettings)
    color: Optional[ColorDetectionSettings] = Field(default_factory=ColorDetectionSettings)


class BaseObjectFilters(BaseModel):
    min_conf: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    total_max_area: Union[float, int, str, None] = Field(default=None)
    total_min_area: Union[float, int, str, None] = Field(default=None)
    max_area: Union[float, int, str, None] = Field(default=None)
    min_area: Union[float, int, str, None] = Field(default=None)

    # validators
    _normalize_areas = field_validator(
        "total_max_area", "total_min_area", "max_area", "min_area"
    )(validate_percentage_or_pixels)


class OverRideObjectFilters(BaseObjectFilters):
    pattern: Optional[Pattern] = None
    labels: Optional[
        Dict[str, Union[BaseObjectFilters, "OverRideObjectFilters", Dict]]
    ] = None


class ObjectFilters(BaseObjectFilters):
    pattern: Optional[Pattern] = None
    labels: Optional[Dict[str, Union[BaseObjectFilters, OverRideObjectFilters]]] = None


class FaceFilters(BaseModel):
    pattern: Optional[Pattern] = Field(default=re.compile("(.*)"))


class OverRideFaceFilters(BaseModel):
    pattern: Optional[Pattern] = None


class AlprFilters(BaseModel):
    pattern: Optional[Pattern] = Field(default=re.compile("(.*)"))
    min_conf: Optional[float] = Field(ge=0.0, le=1.0, default=0.35)


class OverRideAlprFilters(BaseModel):
    pattern: Optional[Pattern] = None
    min_conf: Optional[float] = Field(ge=0.0, le=1.0, default=None)


class StaticObjects(DefaultEnabled):
    enabled: Optional[bool] = Field(False)
    difference: Optional[Union[float, int]] = Field(0.1)
    labels: Optional[List[str]] = Field(default_factory=list)
    ignore_labels: Optional[List[str]] = Field(default_factory=list)

    _validate_difference = field_validator("difference")(validate_percentage_or_pixels)


class OverRideStaticObjects(BaseModel):
    enabled: Optional[bool] = None
    difference: Optional[Union[float, int]] = None
    labels: Optional[List[str]] = None
    ignore_labels: Optional[List[str]] = None

    _validate_difference = field_validator("difference", mode="before")(
        validate_percentage_or_pixels
    )


class MatchFilters(BaseModel):
    object: Optional[ObjectFilters] = Field(default_factory=ObjectFilters)
    face: Optional[FaceFilters] = Field(default_factory=FaceFilters)
    alpr: Optional[AlprFilters] = Field(default_factory=AlprFilters)


class OverRideMatchFilters(BaseModel):
    object: Optional[OverRideObjectFilters] = Field(
        default_factory=OverRideObjectFilters
    )
    face: Optional[OverRideFaceFilters] = Field(default_factory=OverRideFaceFilters)
    alpr: Optional[OverRideAlprFilters] = Field(default_factory=OverRideAlprFilters)


class MatchStrategy(str, Enum):
    # first match wins
    first = "first"
    most = "most"
    most_models = "most_models"
    most_unique = "most_unique"


class MatchingSettings(BaseModel):
    strategy: MatchStrategy = Field(MatchStrategy.first)
    static_objects: StaticObjects = Field(default_factory=StaticObjects)
    filters: MatchFilters = Field(default_factory=MatchFilters)


class MonitorZones(BaseModel):
    enabled: bool = Field(True)
    points: Optional[List[Tuple[int, int]]] = None
    resolution: Optional[Tuple[int, int]] = None
    object_confirm: Optional[bool] = None
    static_objects: Union[OverRideStaticObjects, StaticObjects, None] = Field(
        default_factory=OverRideStaticObjects
    )
    filters: Union[MatchFilters, OverRideMatchFilters, None] = Field(
        default_factory=OverRideMatchFilters
    )

    imported: bool = False

    __validate_resolution = field_validator("resolution", mode="before")(
        validate_resolution
    )
    __validate_points = field_validator("points", mode="before")(validate_points)


class MonitorsSettings(BaseModel):
    models: Optional[Dict[str, Any]] = Field(default_factory=dict)
    object_confirm: Optional[bool] = None
    static_objects: Optional[OverRideStaticObjects] = Field(
        default_factory=OverRideStaticObjects
    )
    filters: Optional[OverRideMatchFilters] = Field(
        default_factory=OverRideMatchFilters
    )
    skip_imported_zones: Optional[bool] = Field(False)
    zones: Optional[Dict[str, MonitorZones]] = Field(default_factory=dict)


class ZMTag(BaseModel):
    Id: Optional[int] = None
    Name: Annotated[str, Field(..., max_length=64)]
    CreatedDate: Optional[datetime] = None
    CreatedBy: Optional[int] = None
    LastAssignedDate: Optional[datetime] = None

class ZMEventsTags(BaseModel):
    TagId: int
    EventId: int
    AssignedDate: Optional[datetime] = None
    AssignedBy: Optional[int] = None

class ConfigFileModel(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    config_path: Path = Field(Path("/etc/zm"))
    system: SystemSettings = Field(default_factory=SystemSettings)
    zoneminder: ZoneMinderSettings = Field(default_factory=ZoneMinderSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    mlapi: ServerRoute = Field(default_factory=ServerRoute)
    animation: AnimationSettings = Field(default_factory=AnimationSettings)
    notifications: MLNotificationSettings = Field(
        default_factory=MLNotificationSettings
    )
    label_groups: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    detection_settings: DetectionSettings = Field(default_factory=DetectionSettings)
    matching: MatchingSettings = Field(default_factory=MatchingSettings)
    monitors: Optional[Dict[Union[int, str], Union[MonitorsSettings, bool]]] = Field(default_factory=dict)

    _validate_config_path = field_validator("config_path", mode="before")(validate_dir)

    @model_validator(mode="after")
    def _validate_model_after(self):
        """check for url attr in ZMSPullMethod"""

        pass


class ClientEnvVars(BaseSettings):

    zm_conf_dir: Path = Field(
        Path("/etc/zm"),
        description="Path to ZoneMinder config files, Default: /etc/zm",
        validation_alias="ML_CLIENT_ZM_CONF_DIR"
    )
    ml_conf_dir: Optional[Path] = Field(
        None,
        description="Path to ZoMi ML config file directory (client/secrets .yml)",
    )
    client_conf_file: Optional[Path] = Field(
        None,
        description="Absolute path to ZoMi ML CLIENT config file",
        alias="ML_CLIENT_CONF_FILE",
    )

    db: Optional[ZMDBSettings] = Field(
        default_factory=ZMDBSettings,
    )
    api: Optional[ZoneMinderSettings.ZMAPISettings] = Field(
        default_factory=ZoneMinderSettings.ZMAPISettings,
    )

    _validate_client_conf_file = field_validator(
        "client_conf_file", mode="before", check_fields=False
    )(validate_file)
    _validate_zm_conf_dir = field_validator(
        "zm_conf_dir", "ml_conf_dir", mode="before"
    )(validate_dir)


class Result(BaseModel):
    label: str
    confidence: float
    bounding_box: List[int]
    color: Optional[Dict[str, float]] = None

    def __eq__(self, other):
        if not isinstance(other, Result):
            return False
        return (
            self.label == other.label
            and self.confidence == other.confidence
            and self.bounding_box == other.bounding_box
        )

    def __str__(self):
        _c: Optional[str] = None
        if self.color:
            _c = f" [Color: {self.color}]"
        return f"'{self.label}'{_c} ({self.confidence:.2f}) @ {self.bounding_box}"

    def __repr__(self):
        _c: Optional[str] = None
        if self.color:
            _c = f" [Color: {self.color}]"
        return f"<'{self.label}'{_c} ({self.confidence * 100:.2f}%) @ {self.bounding_box}>"


class DetectionResults(BaseModel, arbitrary_types_allowed=True):
    success: bool = Field(...)
    name: str = Field(...)
    type: ModelType = Field(...)
    processor: ModelProcessor = Field(...)
    results: Optional[List[Result]] = Field(None)
    removed_by_filters: Optional[List[Result]] = Field(None, repr=False)

    image: Optional[np.ndarray] = Field(None, repr=False)
    extra_image_data: Optional[Dict[str, Any]] = Field(None, repr=False)

    def get_labels(
        self,
    ) -> Tuple[
        List[Optional[str]], List[Optional[float]], List[Optional[List[Optional[int]]]]
    ]:
        if not self.results or self.results is None:
            return []
        return (
            [r.label for r in self.results],
            [r.confidence for r in self.results],
            [r.bounding_box for r in self.results],
        )


class GlobalConfig(BaseModel, arbitrary_types_allowed=True, extra="allow"):
    api: Union[ZMAPI, None] = None
    db: Union[ZMDB, None] = None
    mid: Optional[int] = None
    config: Optional[ConfigFileModel] = None
    config_file: Union[str, Path, None] = None
    configs_path: Union[str, Path, None] = None
    eid: Optional[int] = None
    mon_name: Optional[str] = None
    mon_post: Optional[int] = None
    mon_pre: Optional[int] = None
    mon_fps: Optional[Decimal] = None
    reason: Optional[str] = None
    notes: Optional[str] = None
    event_path: Optional[Path] = None
    event_cause: Optional[str] = None
    past_event: bool = False
    Event: Optional[Dict] = None
    Frame: Optional[List] = None
    mon_image_buffer_count: Optional[int] = None
    mon_width: Optional[int] = None
    mon_height: Optional[int] = None
    mon_colorspace: Optional[int] = None
    frame_buffer: Optional[Dict] = Field(default_factory=dict)
    user_id: Optional[int] = None

    Environment: Optional[Union[ClientEnvVars]] = None
    imported_zones: list = Field(default_factory=list)
    random: Dict = Field(default_factory=dict)
    static_objects: Dict = Field(default_factory=dict)
