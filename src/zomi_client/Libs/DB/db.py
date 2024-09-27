from __future__ import annotations

import logging
import time
from configparser import ConfigParser, SectionProxy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union, Tuple, TYPE_CHECKING, Any, Dict, List, NamedTuple

from pydantic import SecretStr

from sqlalchemy import MetaData, create_engine, select, Column, Integer, ForeignKey, String, DateTime, delete, insert
from sqlalchemy.dialects.mysql import VARCHAR, TIMESTAMP
from sqlalchemy.engine import Engine, Connection, CursorResult, ResultProxy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base

from ...Log import CLIENT_LOGGER_NAME

if TYPE_CHECKING:
    from ...Models.config import GlobalConfig, ZMDBSettings, ClientEnvVars

logger = logging.getLogger(CLIENT_LOGGER_NAME)
LP = "zmdb:"
g: Optional[GlobalConfig] = None


# pydantic allows ORM classes:
Base = declarative_base()

class EventsTags(Base):
    __tablename__ = 'Events_Tags'
    TagId = Column(Integer, ForeignKey('Tags.Id'), primary_key=True)
    EventId = Column(Integer, ForeignKey('Events.Id'), primary_key=True)
    AssignedDate = Column(TIMESTAMP)
    AssignedBy = Column(String)


class ZMTag(Base):
    __tablename__ = 'Tags'
    Id = Column(Integer, primary_key=True)
    Name = Column(VARCHAR(64))
    CreateDate = Column(TIMESTAMP)
    CreatedBy = Column(Integer)
    LastAssignedDate = Column(TIMESTAMP)


class ZMVersion(NamedTuple):
    major: int
    minor: int
    patch: int


class ZMDB:
    engine: Optional[Engine]
    connection: Optional[Connection]
    meta: Optional[MetaData]
    connection_str: str
    config: ZMDBSettings
    conf_file_data: Optional[SectionProxy]
    env: Optional[ClientEnvVars]
    cgi_path: str

    def __init__(self, env: Optional[ClientEnvVars] = None):
        global g
        from ...main import get_global_config

        g = get_global_config()

        # TODO: integrate better
        if env:
            self.env = env
        else:
            self.env = g.Environment

        # logger.debug(f"{LP} ClientEnvVars = {self.env}")

        self.engine: Optional[Engine] = None
        self.connection: Optional[Connection] = None
        self.meta: Optional[MetaData] = None
        g.db = self
        self.config = self.init_config()
        self._db_create()

    def set_config(self, config: ZMDBSettings):
        self.config = config
        self.reset_db()
        self._db_create()

    async def get_all_event_data(self, eid):
        if g.Event is None:
            g.Event = {}
        g.Frame = self.event_frames_data(eid)
        g.Event["MonitorId"] = self.mid_from_eid(eid)
        g.Event["StorageId"] = self.storage_id_from_eid(eid)
        g.Event['Length'] = self.length_from_eid(eid)

    def get_zm_version(self) -> ZMVersion:
        """"""
        _select: select = select(self.meta.tables["Config"].c.Value).where(
            self.meta.tables["Config"].c.Name == "ZM_DYN_CURR_VERSION"
        )
        result: CursorResult = self.run_select(_select)
        for row in result:
            zm_version = row[0]

        zm_version = zm_version.split(".")
        nt = ZMVersion
        zm_version = nt(major=int(zm_version[0]), minor=int(zm_version[1]), patch=int(zm_version[2]))
        return zm_version

    def get_db_version(self):
        _select: select = select(self.meta.tables["Config"].c.Value).where(
            self.meta.tables["Config"].c.Name == "ZM_DYN_DB_VERSION"
        )
        result: CursorResult = self.run_select(_select)
        for row in result:
            db_version = row[0]
        return db_version

    def set_notes(self, eid: int, notes: str):
        _update = self.meta.tables["Events"].update().where(
            self.meta.tables["Events"].c.Id == eid
        ).values(Notes=notes)
        self.connection.execute(_update)
        self.connection.commit()

    def get_tags(self):
        lp = f"{LP}get_tags:"
        _select: select = select(ZMTag)
        result: CursorResult = self.run_select(_select)
        for row in result:
            logger.debug(f"{lp} {row = }")

        return result

    def get_event_tags(self, eid: int):
        lp = f"{LP}get_event_tags:"
        _select: select = select(EventsTags).where(EventsTags.EventId == eid)
        result: CursorResult = self.run_select(_select)
        for row in result:
            logger.debug(f"{lp} {row = }")
        return result

    def set_event_tags(self, eid: int, tags: List[ZMTag]):
        """
        Delete, then insert, tags should be a list of Tag objects.  Since Tags have AssignedBy and AssignedTime etc,
        one should be careful not to lose that data.
        """
        # delete the existing tags? What if other tags are already set by the user (i.e. past event)?

        self.connection.execute(
            delete(EventsTags).where(EventsTags.EventId == eid)
        )
        for tag in tags:
            _insert = insert(EventsTags).values(EventId=eid, TagId=tag.Id, AssignedBy=None, AssignedDate=datetime.timestamp())
            self.connection.execute(_insert)
        self.connection.commit()


    def event_frames_len(self) -> Optional[int]:
        _select: select = select(self.meta.tables["Events"].c.Frames).where(
            self.meta.tables["Events"].c.Id == g.eid
        )
        result: CursorResult = self.run_select(_select)
        for row in result:
            g.Event["Frames"] = row[0]
        return g.Event["Frames"]

    def length_from_eid(self, eid: int) -> Optional[float]:
        _select: select = select(self.meta.tables["Events"].c.Length).where(
            self.meta.tables["Events"].c.Id == eid
        )
        result: CursorResult = self.run_select(_select)
        x: Optional[float] = None
        for row in result:
            x = row[0]
        return x

    def end_datetime_from_eid(self, eid: int):
        _select: select = select(self.meta.tables["Events"].c.EndDateTime).where(
            self.meta.tables["Events"].c.Id == eid
        )
        result: CursorResult = self.run_select(_select)
        for row in result:
            g.Event["EndDateTime"] = row[0]
        if g.Event["EndDateTime"] is None or not g.Event["EndDateTime"]:
            g.Event["EndDateTime"] = False
        return g.Event["EndDateTime"]

    def get_snapshot_fid(self):
        _select: select = select(self.meta.tables["Events"].c.MaxScoreFrameId).where(
            self.meta.tables["Events"].c.Id == g.eid
        )
        result: CursorResult = self.run_select(_select)
        for row in result:
            g.Event["MaxScoreFrameId"] = row[0]
        return g.Event["MaxScoreFrameId"]

    @staticmethod
    def _rel_path(
        eid: int, mid: int, scheme: str, dt: Optional[datetime] = None
    ) -> str:
        ret_val: str = ""
        lp: str = f"{LP}relative path::"
        if scheme == "Deep":
            if dt:
                ret_val = f"{mid}/{dt.strftime('%y/%m/%d/%H/%M/%S')}"
            else:
                logger.error(f"{lp} no datetime for deep scheme path!")
        elif scheme == "Medium":
            ret_val = f"{mid}/{dt.strftime('%Y-%m-%d')}/{eid}"
        elif scheme == "Shallow":
            ret_val = f"{mid}/{eid}"
        else:
            logger.error(f"{lp} unknown scheme {scheme}")
        return ret_val

    def read_zm_configs(self):
        files = []
        conf_path = g.config.zoneminder.conf_dir
        if not conf_path:
            logger.debug(f"{LP} no ZM .conf files found in config file (zoneminder>conf_dir), checking ENV vars")
            conf_path = self.env.zm_conf_dir
        if conf_path.is_dir():
            for fi in Path(f"{conf_path}/conf.d").glob("*.conf"):
                files.append(fi)
            files.sort()
            files.insert(0, f"{conf_path}/zm.conf")
            config_file = ConfigParser(interpolation=None, inline_comment_prefixes="#")
            try:
                for f in files:
                    with open(f, "r") as zm_conf_file:
                        # This adds [zm_root] section to the head of each zm .conf.d config file,
                        # not physically only in memory
                        _data = zm_conf_file.read()
                        config_file.read_string(f"[zm_root]\n{_data}")
            except Exception as exc:
                logger.error(f"{LP} error opening ZoneMinder .conf files: {exc}")
            else:
                # logger.debug(f"{LP} ZoneMinder .conf files -> {files}")
                # for section in config_file.sections():
                #     for key, value in config_file.items(section):
                #         logger.debug(f"{section} >>> {key} = {value}")
                return config_file["zm_root"]

    def init_config(self):
        """Initialize ZMDBSettings using ENV, zm .conf files and finally internal defaults"""
        defaults = {
            "host": "localhost",
            "port": 3306,
            "user": "zmuser",
            "password": "zmpass",
            "name": "zm",
            "driver": "mysql+pymysql",
        }

        self.conf_file_data = self.read_zm_configs()
        self.cgi_path = self.conf_file_data.get(
            "zm_path_cgi", "COULDN'T GET: ZM_PATH_CGI"
        )
        # ZM_PATH_CGI=/usr/lib/zoneminder/cgi-bin
        _pydantic_attrs = [
            "construct",
            "copy",
            "dict",
            "from_orm",
            "json",
            "p",
            "parse_file",
            "parse_obj",
            "parse_raw",
            "schema",
            "schema_json",
            "update_forward_refs",
            "validate",
        ]
        _pydantic_v2_attrs = [
            "model_computed_fields",
            "model_config",
            "model_construct",
            "model_copy",
            "model_dump",
            "model_dump_json",
            "model_extra",
            "model_fields",
            "model_fields_set",
            "model_json_schema",
            "model_parametrized_name",
            "model_post_init",
            "model_rebuild",
            "model_validate",
            "model_validate_json",
            "settings_customise_sources",
        ]
        _pydantic_attrs.extend(_pydantic_v2_attrs)
        db_config_with_env = self.env.db
        logger.debug(f"{LP} ENV VARS = {db_config_with_env}")
        for _attr in dir(db_config_with_env):
            if _attr in ["host", "port", "user", "password", "name", "driver"]:
                # check the env
                if not (set_to := getattr(db_config_with_env, _attr)):
                    # env var is not set, try to get attr from ZM .conf files
                    xtra_ = ""
                    unset_ = ""
                    conf_val = f"ZM_DB_{_attr.upper()}"
                    if _attr == "password":
                        conf_val = f"ZM_DB_PASS"

                    unset_ += "ENV "
                    if conf_val in self.conf_file_data:
                        set_to = (
                            self.conf_file_data[conf_val]
                            if _attr != "password"
                            else SecretStr(self.conf_file_data[conf_val])
                        )
                        xtra_ = f" (defaulting to '{set_to}' from ZM .conf files)"

                # check the config file
                if g and g.config:
                    if g.config.zoneminder.db:
                        cfg_file_db = getattr(g.config.zoneminder.db, _attr)
                        if cfg_file_db is not None:
                            # There is an entry in the config file, use it even if EN-V or .conf files set it
                            set_to = cfg_file_db
                            xtra_ = f" (OVERRIDING to '{set_to}' from zomi config file)"

                if not set_to:
                    # not in env, ZM .conf files or config file try internal defaults
                    unset_ += "CFG .CONFs "
                    set_to = defaults[_attr]
                    xtra_ = f" (defaulting to '{set_to}' from internal defaults)"
                logger.debug(f"{LP} [{unset_.rstrip()}] unset for db. {_attr}{xtra_}")
                setattr(db_config_with_env, _attr, set_to)
        return db_config_with_env

    def _db_create(self):
        """A private function to interface with the ZoneMinder DataBase"""
        # From @pliablepixels SQLAlchemy work - all credit goes to them.
        lp: str = f"{LP}init::"
        _pw = (
            self.config.password.get_secret_value()
            if isinstance(self.config.password, SecretStr)
            else self.config.password
        )
        self.connection_str = (
            f"{self.config.driver}://{self.config.user}"
            f":{_pw}@{self.config.host}"
            f"/{self.config.name}"
        )
        self._check_conn()

    def _check_conn(self):
        """A private function to create the DB engine, connection and metadata if not already created"""
        try:
            if not self.engine:
                # logger.debug(f"{LP} creating engine with {self.connection_str = } TYPE={type(self.connection_str)}")
                self.engine = create_engine(self.connection_str, pool_recycle=3600)
            if not self.connection:
                self.connection = self.engine.connect()
            if not self.meta:
                self._refresh_meta()
        except SQLAlchemyError as e:
            logger.error(
                f"{self.connection_str = } :: TYPE={type(self.connection_str)}"
            )
            logger.error(f"Could not connect to DB, message was: {e}")
            raise e
        except Exception as e:
            logger.error(
                f"Exception while checking DB connection on _check_conn() -> {e}"
            )
            raise e

    def _refresh_meta(self):
        """A private function to refresh the DB metadata"""
        del self.meta
        self.meta = None
        self.meta = MetaData()
        self.meta.reflect(
            bind=self.engine,
            only=["Events", "Monitors", "Monitor_Status", "Storage", "Frames", "Config", "Zones", "Tags", "Events_Tags"],
        )

    def run_select(self, select_stmt: select) -> ResultProxy:
        """A method to run a select statement"""
        self._check_conn()
        try:
            result = self.connection.execute(select_stmt)
        except SQLAlchemyError as e:
            logger.error(f"Could not read from DB, message was: {e}")
        else:
            return result

    def mid_from_eid(self, eid: int) -> int:
        """A method to get the Monitor ID from the Event ID"""
        mid: int = 0
        e_select: select = select(self.meta.tables["Events"].c.MonitorId).where(
            self.meta.tables["Events"].c.Id == eid
        )
        mid_result: CursorResult = self.run_select(e_select)
        for row in mid_result:
            mid = row[0]
        mid_result.close()
        return int(mid)

    def event_frames_data(self, eid: int):
        """A method to get the event frames data from the DB for a given event ID"""
        event_frames: List[Dict[str, Any]] = []
        _select: select = select(self.meta.tables["Frames"]).where(
            self.meta.tables["Frames"].c.EventId == eid
        )
        result: CursorResult = self.run_select(_select)
        for row in result:
            fdict = {}
            fdict["Id"] = row[0]
            fdict["EventId"] = row[1]
            fdict["FrameId"] = row[2]
            fdict["Type"] = row[3]
            fdict["TimeStamp"] = row[4].strftime("%Y-%m-%d %H:%M:%S")
            fdict["Delta"] = float(row[5])
            fdict["Score"] = row[6]
            fdict["TimeStampSecs"] = int(datetime.timestamp(row[4]))
            event_frames.append(fdict)
        result.close()
        # logger.debug(f"{LP} {event_frames = }")
        return event_frames

    def _mon_name_from_mid(self, mid: int) -> str:
        """Get the monitor name from the DB."""
        mon_name = None
        mid_name_select: select = select(self.meta.tables["Monitors"].c.Name).where(
            self.meta.tables["Monitors"].c.Id == mid
        )
        mid_name_result: CursorResult = self.run_select(mid_name_select)
        for mon_row in mid_name_result:
            mon_name = mon_row[0]
        mid_name_result.close()
        return mon_name

    def mon_preBuffer_from_mid(self, mid: int) -> int:
        mon_pre: int = 0
        pre_event_select: select = select(
            self.meta.tables["Monitors"].c.PreEventCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)
        result: CursorResult = self.connection.execute(pre_event_select)
        for mon_row in result:
            mon_pre = mon_row[0]
        result.close()
        return int(mon_pre)

    def mon_postBuffer_from_mid(self, mid: int) -> int:
        mon_post: int = 0
        post_event_select: select = select(
            self.meta.tables["Monitors"].c.PostEventCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)
        select_result: CursorResult = self.connection.execute(post_event_select)

        for mon_row in select_result:
            mon_post = mon_row[0]
        select_result.close()
        return int(mon_post)

    def zones_from_mid(self, mid: int) -> List[Dict[str, Any]]:
        zones: List[Dict[str, Any]] = []
        _select: select = select(self.meta.tables["Zones"]).where(
            self.meta.tables["Zones"].c.MonitorId == mid
        )
        result: CursorResult = self.run_select(_select)
        for row in result:
            zdict = {}
            zdict["Id"] = row[0]
            zdict["MonitorId"] = row[1]
            zdict["Name"] = row[2]
            zdict["Type"] = row[3]
            zdict["Units"] = row[4]
            zdict["NumCoords"] = row[5]
            zdict["Coords"] = row[6]
            zdict["Area"] = row[7]
            zdict["AlarmRGB"] = row[8]
            zdict["CheckMethod"] = row[9]
            zdict["MinPixelThreshold"] = row[10]
            zdict["MaxPixelThreshold"] = row[11]
            zdict["MinAlarmPixels"] = row[12]
            zdict["MaxAlarmPixels"] = row[13]
            zdict["FilterX"] = row[14]
            zdict["FilterY"] = row[15]
            zdict["MinFilterPixels"] = row[16]
            zdict["MaxFilterPixels"] = row[17]
            zdict["MinBlobPixels"] = row[18]
            zdict["MaxBlobPixels"] = row[19]
            zdict["MinBlobs"] = row[20]
            zdict["MaxBlobs"] = row[21]
            zdict["OverloadFrames"] = row[22]
            zdict["ExtendAlarmFrames"] = row[22]

            zones.append(zdict)
        result.close()
        return zones

    async def import_zones(self):
        """A method to import zones that are defined in the ZoneMinder web GUI instead of defining
        zones in the per-monitor section of the configuration file.


        :return:
        """
        from ...Models.config import MonitorsSettings

        imported_zones: List = []
        lp: str = f"{LP}import zones::"
        mid_cfg: Optional[MonitorsSettings] = None
        existing_zones: Dict = {}
        if g.config.detection_settings.import_zones:
            from ...Models.config import MonitorZones

            mid_cfg = g.config.monitors.get(g.mid)
            if mid_cfg:
                existing_zones: Dict = mid_cfg.zones
            if existing_zones is None:
                existing_zones = {}
            monitor_resolution: Tuple[int, int] = (int(g.mon_width), int(g.mon_height))
            zones = self.zones_from_mid(g.mid)
            if zones:
                logger.debug(
                    f"{lp} {len(zones)} ZM zones found, checking for 'Inactive'/'Private' zones"
                )
                for zone in zones:
                    zone_name: str = zone.get("Name", "")
                    zone_type: str = zone.get("Type", "")
                    zone_points: str = zone.get("Coords", "")
                    # logger.debug(f"{lp} BEGINNING OF ZONE LOOP - {zone_name=} -- {zone_type=} -- {zone_points=}")
                    if zone_type.casefold() in [
                        "inactive",
                        "privacy",
                        "preclusive",
                    ]:
                        logger.debug(
                            f"{lp} skipping '{zone_name}' as it is set to '{zone_type.capitalize()}'"
                        )
                        continue

                    if not mid_cfg:
                        logger.debug(
                            f"{lp} no monitor configuration found for monitor {g.mid}, "
                            f"creating a new one and adding zone '{zone_name}' as first entry"
                        )
                        mid_cfg = MonitorsSettings(
                            models=None,
                            object_confirm=None,
                            static_objects=None,
                            filters=None,
                            zones={
                                zone_name: MonitorZones(
                                    points=zone_points,
                                    resolution=monitor_resolution,
                                    imported=True,
                                )
                            },
                        )
                        g.config.monitors[g.mid] = mid_cfg
                        existing_zones = mid_cfg.zones
                        continue

                    if mid_cfg:
                        # logger.debug(f"{lp} existing zones found: {existing_zones}")
                        if not (existing_zone := existing_zones.get(zone_name)):
                            logger.debug(
                                f"{lp} Imported zone->'{zone_name}' is "
                                f"being converted into a ML zone for monitor {g.mid}"
                            )
                            new_zone = MonitorZones(
                                points=zone_points,
                                resolution=monitor_resolution,
                                imported=True,
                            )
                            g.config.monitors[g.mid].zones[zone_name] = new_zone
                            imported_zones.append({zone_name: new_zone})

                        else:
                            logger.debug(
                                f"{lp} '{zone_name}' is defined in zomi-client monitor {g.mid} configuration"
                            )
                            # only update if points are not set
                            if not existing_zone.points:
                                # logger.debug(f"{lp} updating points for '{zone_name}'")
                                ex_z_dict = existing_zone.dict()
                                # logger.debug(f"{lp} existing zone AS DICT: {ex_z_dict}")
                                ex_z_dict["points"] = zone_points
                                ex_z_dict["resolution"] = monitor_resolution
                                # logger.debug(f"{lp} updated zone AS DICT: {ex_z_dict}")
                                existing_zone = MonitorZones(**ex_z_dict)
                                # logger.debug(f"{lp} updated zone AS MODEL: {existing_zone}")
                                g.config.monitors[g.mid].zones[
                                    zone_name
                                ] = existing_zone
                                imported_zones.append({zone_name: existing_zone})
                                logger.debug(
                                    f"{lp} '{zone_name}' is defined in the config, updated points and resolution to match ZM"
                                )
                            else:
                                logger.warning(
                                    f"{lp} '{zone_name}' HAS POINTS SET which is interpreted "
                                    f"as a ML configured zone, not importing ZM defined zone points"
                                )
                    # logger.debug(f"{lp}DBG>>> END OF ZONE LOOP for '{zone_name}'")
            else:
                logger.debug(f"{lp} no ZM defined zones found for monitor {g.mid}")
        else:
            logger.debug(f"{lp} import_zones() is disabled, skipping")
        # logger.debug(f"{lp} ALL ZONES with imported zones => {imported_zones}")
        return imported_zones
        
    def mon_fps_from_mid(self, mid: int) -> Decimal:
        mon_fps: Decimal = Decimal(0)
        # Get Monitor capturing FPS
        ms_select: select = select(
            self.meta.tables["Monitor_Status"].c.CaptureFPS
        ).where(self.meta.tables["Monitor_Status"].c.MonitorId == mid)

        select_result: CursorResult = self.run_select(ms_select)
        for mons_row in select_result:
            mon_fps = float(mons_row[0])
        select_result.close()
        return Decimal(mon_fps)

    def _reason_from_eid(self, eid: int) -> str:
        reason: str = ""
        reason_select: select = select(self.meta.tables["Events"].c.Cause).where(
            self.meta.tables["Events"].c.Id == eid
        )
        reason_result: CursorResult = self.connection.execute(reason_select)
        for row in reason_result:
            reason = row[0]
        reason_result.close()
        return reason

    def event_notes(self, eid: int) -> str:
        notes: str = ""
        notes_select: select = select(self.meta.tables["Events"].c.Notes).where(
            self.meta.tables["Events"].c.Id == eid
        )
        notes_result: CursorResult = self.connection.execute(notes_select)
        for row in notes_result:
            notes = row[0]
        notes_result.close()
        return notes

    def _scheme_from_eid(self, eid: int):
        scheme = None
        scheme_select: select = select(self.meta.tables["Events"].c.Scheme).where(
            self.meta.tables["Events"].c.Id == eid
        )
        scheme_result: CursorResult = self.connection.execute(scheme_select)
        for row in scheme_result:
            scheme = row[0]
        scheme_result.close()
        return scheme

    def storage_id_from_eid(self, eid: int) -> int:
        storage_id: Optional[int] = None
        storage_id_select: select = select(
            self.meta.tables["Events"].c.StorageId
        ).where(self.meta.tables["Events"].c.Id == eid)
        storage_id_result: CursorResult = self.run_select(storage_id_select)
        for row in storage_id_result:
            storage_id = row[0]
        storage_id_result.close()
        if storage_id is not None:
            storage_id = (
                1 if storage_id == 0 else storage_id
            )  # Catch 0 and treat as 1 (zm code issue)
        return storage_id

    def start_datetime_from_eid(self, eid: int) -> datetime:
        start_datetime: Optional[datetime] = None
        start_datetime_select: select = select(
            self.meta.tables["Events"].c.StartDateTime
        ).where(self.meta.tables["Events"].c.Id == eid)
        start_datetime_result: CursorResult = self.connection.execute(
            start_datetime_select
        )
        for row in start_datetime_result:
            start_datetime = row[0]
        start_datetime_result.close()
        return start_datetime

    def _storage_path_from_storage_id(self, storage_id: int) -> str:
        storage_path: str = ""
        storage_path_select: select = select(self.meta.tables["Storage"].c.Path).where(
            self.meta.tables["Storage"].c.Id == storage_id
        )
        storage_path_result: CursorResult = self.connection.execute(storage_path_select)
        for row in storage_path_result:
            storage_path = row[0]
        storage_path_result.close()
        return storage_path

    def _get_mon_shape_from_mid(self, mid: int) -> Tuple[int, int, int]:
        """Get the monitor shape from the DB. (W, H, C)"""
        width: int = 0
        height: int = 0
        color: int = 0
        width_select: select = select(self.meta.tables["Monitors"].c.Width).where(
            self.meta.tables["Monitors"].c.Id == mid
        )
        height_select: select = select(self.meta.tables["Monitors"].c.Height).where(
            self.meta.tables["Monitors"].c.Id == mid
        )
        colours_select: select = select(self.meta.tables["Monitors"].c.Colours).where(
            self.meta.tables["Monitors"].c.Id == mid
        )

        width_result: CursorResult = self.run_select(width_select)
        for mon_row in width_result:
            width = mon_row[0]
        width_result.close()
        # height
        height_result: CursorResult = self.run_select(height_select)
        for mon_row in height_result:
            height = mon_row[0]
            g.mon_height = height
        height_result.close()
        # colours
        colours_result: CursorResult = self.run_select(colours_select)
        for mon_row in colours_result:
            color = mon_row[0]
        colours_result.close()
        return width, height, color

    def _get_image_buffer_from_mid(self, mid: int) -> int:
        """Get the monitor ImageBufferCount from the DB.

        Key in DB: 'ImageBufferCount'"""
        buffer: Optional[int] = None
        buffer_select: select = select(
            self.meta.tables["Monitors"].c.ImageBufferCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)
        buffer_result: CursorResult = self.connection.execute(buffer_select)
        for mon_row in buffer_result:
            buffer = mon_row[0]
        buffer_result.close()
        return buffer

    def cause_from_eid(self, eid: int) -> str:
        """Get the cause of the event from the DB.

        Key in DB: 'Cause'"""
        return self._reason_from_eid(eid)

    def eid_exists(self, eid: int) -> bool:
        """Check if an event ID exists in the DB

        Key in DB: 'Id'"""
        event_exists: bool = False
        event_exists = (
            self.run_select(
                select(self.meta.tables["Events"]).where(
                    self.meta.tables["Events"].c.Id == eid
                )
            ).fetchone()
            is not None
        )
        return event_exists

    def grab_all(self, eid: int) -> Tuple[int, str, int, int, Decimal, str, str]:
        # FIX ME!!!! A hammer to grab all the data from the DB for a given event ID
        if g.Event is None:
            g.Event = {}
        _start = time.time()
        event_exists: bool = self.eid_exists(eid)
        if not event_exists:
            raise ValueError(f"Event ID {eid} does not exist in ZoneMinder DB")

        storage_path: Optional[str] = None
        event_path: Optional[Union[Path, str]] = None
        mid: Optional[Union[str, int]] = self.mid_from_eid(eid)
        mon_name: Optional[str] = self._mon_name_from_mid(mid)
        mon_post: Optional[Union[str, int]] = self.mon_postBuffer_from_mid(mid)
        mon_pre: Optional[Union[str, int]] = self.mon_preBuffer_from_mid(mid)
        mon_fps: Optional[Union[float, Decimal]] = self.mon_fps_from_mid(mid)
        reason: Optional[str] = self._reason_from_eid(eid)
        notes: Optional[str] = self.event_notes(eid)
        scheme: Optional[str] = self._scheme_from_eid(eid)
        storage_id: Optional[int] = self.storage_id_from_eid(eid)
        start_datetime: Optional[datetime] = self.start_datetime_from_eid(eid)
        end_datetime: Optional[datetime] = self.end_datetime_from_eid(eid)
        height, width, color = self._get_mon_shape_from_mid(mid)
        ring_buffer: Optional[int] = self._get_image_buffer_from_mid(mid)
        if storage_id:
            storage_path = self._storage_path_from_storage_id(storage_id)

        final_str: str = ""
        if mid:
            final_str += f"Monitor ID: {mid} "
        else:
            raise ValueError("No Monitor ID returned from DB query")
        if mon_name:
            final_str += f"Monitor Name: {mon_name} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor name ('Name') for monitor ID {mid}"
            )
        if mon_pre:
            final_str += f"Monitor PreEventCount: {mon_pre} "
        else:
            logger.warning(
                f"{LP} the database query did not return monitor pre-event count ('PreEventCount') for monitor ID {mid}"
            )
        if mon_post:
            final_str += f"Monitor PostEventCount: {mon_post} "
        else:
            logger.warning(
                f"{LP} the database query did not return monitor post-event count ('PostEventCount') for monitor ID {mid}"
            )
        if mon_fps:
            final_str += f"Monitor FPS: {mon_fps} "
        else:
            logger.warning(
                f"{LP} the database query did not return monitor FPS ('CaptureFPS') for monitor ID {mid}"
            )
        if reason:
            final_str += f"Event Cause: {reason} "
        else:
            logger.warning(
                f"{LP} the database query did not return a 'reason' ('Cause') for this event!"
            )
        if notes:
            final_str += f"Event Notes: {notes} "
        else:
            logger.warning(
                f"{LP} the database query did not return any notes ('Notes') for this event!"
            )
        if scheme:
            final_str += f"Event Storage Scheme: {scheme} "
        else:
            logger.warning(
                f"{LP} the database query did not return any scheme ('Scheme') for this event!"
            )
        if storage_id:
            final_str += f"Event Storage ID: {storage_id} "
        else:
            logger.warning(
                f"{LP} the database query did not return any storage ID ('StorageId') for this event!"
            )
        if start_datetime:
            final_str += f"Event StartDateTime: {start_datetime} "
        else:
            logger.warning(
                f"{LP} the database query did not return any start datetime ('StartDateTime') for this event!"
            )
        if end_datetime:
            final_str += f"Event EndDateTime: {end_datetime} "
        else:
            logger.warning(
                f"{LP} the database query did not return any end datetime ('EndDateTime') for this event!"
            )
        if storage_path:
            final_str += f"Event Storage Path: {storage_path} "
        else:
            logger.warning(
                f"{LP} the database query did not return a storage path ('Path') for this event!"
            )
        if width:
            final_str += f"Monitor Width: {width} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor width ('Width') for this event!"
            )
        if height:
            final_str += f"Monitor Height: {height} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor height ('Height') for this event!"
            )
        if color:
            final_str += f"Monitor Color: {color} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor color ('Colours') for this event!"
            )
        if ring_buffer:
            final_str += f"Monitor ImageBufferCount: {ring_buffer}"
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor ImageBufferCount ('ImageBufferCount') for this event!"
            )
        if storage_path:
            if eid and mid and scheme and start_datetime:
                event_path = Path(
                    f"{storage_path}/{self._rel_path(eid, mid, scheme, start_datetime)}"
                )
            else:
                logger.error(
                    f"{LP} no event ID ({eid}), monitor ID ({mid}), scheme ({scheme}) or start_datetime ({start_datetime}) to calculate the storage path!"
                )
        else:
            if storage_id:
                logger.error(
                    f"{LP} no storage path for StorageId {storage_id}, the StorageId could "
                    f"of been removed/deleted/disabled"
                )
            else:
                logger.error(
                    f"{LP} no StorageId for event {eid} to calculate the storage path!"
                )

        if event_path:
            logger.debug(
                f"{LP} storage path for event ID: {eid} has been calculated as '{event_path}'"
            )
        else:
            logger.warning(f"{LP} could not calculate the storage path for this event!")

        logger.debug(
            f"perf:{LP} Grabbing DB info took {time.time() - _start:.5f} s"
        )
        return (
            mid,
            mon_name,
            mon_post,
            mon_pre,
            mon_fps,
            reason,
            event_path,
            notes,
            width,
            height,
            color,
            ring_buffer,
        )

    def reset_db(self):
        if self.engine:
            self.engine.dispose()
            self.engine = None
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.meta:
            self.meta.clear()
            self.meta = None

    def clean_up(self):
        if self.connection.closed is False:
            self.connection.close()
            logger.debug(f"{LP}close: closed connection to ZoneMinder DB")
        else:
            logger.debug(f"{LP}close: ZoneMinder DB connection already closed")
