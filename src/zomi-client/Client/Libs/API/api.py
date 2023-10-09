from __future__ import annotations

import asyncio
import datetime
import logging
import pickle
import time
from typing import Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import warnings
from ...Log import CLIENT_LOGGER_NAME
logger: logging.Logger = logging.getLogger(CLIENT_LOGGER_NAME)

try:
    import aiohttp
    from jose import jwt
    from pydantic import SecretStr
    from requests import Response, Session
    from requests.exceptions import HTTPError, JSONDecodeError
    from urllib3 import disable_warnings
    from urllib3.exceptions import InsecureRequestWarning
except ImportError as e:
    msg = f"ImportError: {e}"
    logger.critical(msg)
    warnings.warn(msg, ImportWarning)
    print(msg)
    aiohttp: Optional[aiohttp] = None
    jwt: Optional[jwt] = None
    SecretStr: Optional[SecretStr] = None
    Response: Optional[Response] = None
    Session: Optional[Session] = None
    HTTPError: Optional[HTTPError] = None
    JSONDecodeError: Optional[JSONDecodeError] = None
    disable_warnings: Optional[disable_warnings] = None
    InsecureRequestWarning: Optional[InsecureRequestWarning] = None



from ...Models.config import ZoneMinderSettings, MonitorsSettings

if TYPE_CHECKING:
    from ....Shared.configs import GlobalConfig
    import aiohttp
    from pydantic import SecretStr
    from requests import Response, Session
    from requests.exceptions import HTTPError, JSONDecodeError
    from urllib3 import disable_warnings
    from urllib3.exceptions import InsecureRequestWarning

GRACE: int = 60 * 5  # 5 mins
lp: str = "api::"

g: Optional[GlobalConfig] = None
LP: str = "api::"


class ZMAPI:
    async def import_zones(self):
        """A function to import zones that are defined in the ZoneMinder web GUI instead of defining
        zones in the per-monitor section of the configuration file.


        :return:
        """
        imported_zones: List = []
        lp: str = "api::import zones::"
        mid_cfg: Optional[MonitorsSettings] = None
        existing_zones: Dict = {}
        if g.config.detection_settings.import_zones:
            mid_cfg = g.config.monitors.get(g.mid)
            if mid_cfg:
                existing_zones: Dict = mid_cfg.zones
            monitor_resolution: Tuple[int, int] = (int(g.mon_width), int(g.mon_height))
            url = f"{self.portal_base_url}/api/zones/forMonitor/{g.mid}.json"
            # This should be a JSON response parsed into a dictionary
            r: Dict = await self.make_async_request(url)
            if r:
                # logger.debug(f"{lp} RESPONSE from zone API call => {r}")
                zones = r.get("zones")
                if zones:
                    logger.debug(
                        f"{lp} {len(zones)} ZM zones found, checking for 'Inactive'/'Private' zones"
                    )
                    for zone in zones:
                        zone_name: str = zone.get("Zone", {}).get("Name", "")
                        zone_type: str = zone.get("Zone", {}).get("Type", "")
                        zone_points: str = zone.get("Zone", {}).get("Coords", "")
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

                        from ...Models.config import MonitorZones

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
                                    )
                                },
                            )
                            g.config.monitors[g.mid] = mid_cfg
                            existing_zones = mid_cfg.zones
                            continue

                        if existing_zones is None:
                            existing_zones = {}

                        if mid_cfg:
                            if existing_zones is not None:
                                # logger.debug(f"{lp} existing zones found: {existing_zones}")
                                if not (existing_zone := existing_zones.get(zone_name)):
                                    logger.debug(
                                        f"{lp} Zone->'{zone_name}' is being constructed into a model and imported into monitor {g.mid} config"
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
                                        f"{lp} '{zone_name}' is defined in ZM ML monitor {g.mid} configuration -> {existing_zone}"
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
                                        imported_zones.append(
                                            {zone_name: existing_zone}
                                        )
                                        logger.debug(
                                            f"{lp} '{zone_name}' already exists, updated points and resolution"
                                        )
                                    else:
                                        logger.warning(
                                            f"{lp} '{zone_name}' HAS POINTS SET which is interpreted "
                                            f"as a ZM ML configured zone, not importing ZM defined zone points"
                                        )
                        # logger.debug(f"{lp}DBG>>> END OF ZONE LOOP for '{zone_name}'")
                else:
                    logger.debug(f"{lp} no ZM defined zones found for monitor {g.mid}")
            else:
                logger.debug(f"{lp} no response from ZM API for monitor {g.mid} zones")
        else:
            logger.debug(f"{lp} import_zones() is disabled, skipping")
        # logger.debug(f"{lp} ALL ZONES with imported zones => {imported_zones}")
        return imported_zones

    def __init__(self, config: ZoneMinderSettings):
        lp: str = f"{LP}init::"
        global g
        from ...main import get_global_config
        from pydantic import SecretStr

        g = get_global_config()
        self.token_file = g.config.system.variable_data_path / "api_access_token"
        self.access_token: Optional[str] = ""
        self.refresh_token: Optional[str] = ""
        self.config: ZoneMinderSettings.ZMAPISettings = config.api
        self.api_url: Optional[str] = config.api.api_url
        self.portal_url: Optional[str] = config.portal_url

        self.api_version: Optional[str] = ""
        self.zm_version: Optional[str] = ""
        self.zm_tz: Optional[str] = None

        self.session: Session = Session()
        self.async_session: Optional[aiohttp.ClientSession] = None
        self.username: Optional[SecretStr] = self.config.user
        self.password: Optional[SecretStr] = self.config.password

        # Sanitize logs of urls, passwords, tokens, etc. Makes for easier copy+paste
        self.sanitize = g.config.logging.sanitize.enabled
        self.sanitize_str: str = g.config.logging.sanitize.replacement_str
        if self.config.ssl_verify is False:
            self.session.verify = False
            logger.warning(
                f"{lp} SSL certificate verification disabled (encryption enabled, vulnerable to MITM attacks)",
            )
            disable_warnings(category=InsecureRequestWarning)

        if self.token_file:
            _ = self.cached_tokens
            self._refresh_tokens_if_needed()
        else:
            self._login()

    def show_portal(self):
        pass

    def get_session(self):
        return self.session

    def tz(self, force=False):
        if not self.zm_tz or self.zm_tz and force:
            url = f"{self.api_url}/host/gettimezone.json"
            try:
                r = self.make_request(url=url)
            except HTTPError as err:
                logger.error(
                    f"{lp} timezone API not found, relative timezones will be local time",
                )
                logger.debug(f"{lp} EXCEPTION>>> {err}")
            else:
                self.zm_tz = r.get("tz")

        return self.zm_tz

    # called in _make_request to avoid 401s if possible
    def _refresh_tokens_if_needed(self, grace_period: Optional[float] = None):
        lp = f"api::_refresh_tokens_if_needed::"
        _login = False
        _relogin = False
        if not self.access_token:
            logger.warning(f"{lp} no access token to evaluate, calling login()")
            _login = True
        else:
            claims = jwt.get_unverified_claims(
                self.access_token,
            )
            iss = claims.get("iss")
            if iss and iss != "ZoneMinder":
                logger.error(
                    f"{lp} invalid 'iss' [{iss}] for access token, calling login()"
                )
                _login = True
            elif not iss:
                logger.error(
                    f"{lp} no 'Issuer' ['iss'] for access token, calling login()"
                )
                _login = True
            iat = claims["iat"]
            exp = claims["exp"]
            now = time.time()
            remaining = exp - now
            m, s = divmod(remaining, 60)
            h, m = divmod(m, 60)
            if not grace_period:
                grace_period = GRACE
            # logger.debug(
            #     f"{lp} GRACE: {grace_period} --  ISSUED AT: {datetime.datetime.fromtimestamp(iat)}"
            #     f" EXPIRES AT: {datetime.datetime.fromtimestamp(exp)} -- CURRENTLY: "
            #     f"{datetime.datetime.fromtimestamp(now)}   -------- remaining seconds = {remaining}"
            # )
        if not _login and not _relogin:
            if exp > now > (exp - grace_period):
                logger.debug(
                    f"{lp} access token is within {grace_period}s of expiring [{remaining=}], refreshing"
                )
                _relogin = True
            elif now > exp:
                logger.debug(f"{lp} access token is expired [{remaining=}], refreshing")
                _relogin = True
            elif now < iat:
                logger.debug(
                    f"{lp} access token issued in the future?! requesting new tokens"
                )
                _login = True
            else:
                logger.debug(f"{lp} access token is not expired, checking validity")
                # use token to test if it's still valid
                url = f"{self.api_url}/host/getversion.json"
                try:
                    r = self.session.get(url=url, params={"token": self.access_token})
                    r.raise_for_status()
                except HTTPError as err:
                    logger.warning(
                        f"{lp} access token validity test threw an exception! {err}"
                    )
                    logger.debug(f"{lp} EXCEPTION>>> {err}")
                    _login = True
                else:
                    logger.debug(
                        f"{lp} access token is valid for {h:.0f} hours {m:.0f} minutes and {s:.06f} seconds"
                    )
                    return
        if _login:
            logger.debug(f"{lp} calling login()")
            self._login()
        elif _relogin:
            logger.debug(f"{lp} calling re-login()")
            self._re_login()

    def _re_login(self, grace_period: Optional[float] = None, reauth: bool = True):
        lp = f"api::_re_login::"
        _type = "refresh"
        tkn = self.refresh_token
        _login = False
        if not tkn:
            logger.warning(f"{lp} no {_type} token, calling login()")
            _login = True
        if not grace_period:
            grace_period = GRACE
            if not _login:
                claims = jwt.get_unverified_claims(tkn)
                iss = claims.get("iss")
                if iss and iss != "ZoneMinder":
                    logger.error(
                        f"{lp} invalid 'iss' [{iss}] for {_type} token, calling login()"
                    )
                    _login = True
                elif not iss:
                    logger.error(f"{lp} no 'iss' for {_type} token, calling login()")
                    _login = True
                iat = claims["iat"]
                exp = claims["exp"]
                now = time.time()
                remaining = exp - now
                m, s = divmod(remaining, 60)
                h, m = divmod(m, 60)
                if exp > now > (exp - grace_period):
                    logger.debug(
                        f"{lp} {_type} token is within {grace_period} second(s) "
                        f"of expiring, requesting new tokens"
                    )
                    _login = True
                elif now > exp:
                    logger.debug(
                        f"{lp} {_type} token is expired, requesting new tokens"
                    )
                    _login = True
                elif now < iat:
                    logger.debug(
                        f"{lp} {_type} token is in the future, requesting new tokens"
                    )
                    _login = True
                else:
                    logger.debug(
                        f"{lp} {_type} token is valid for {h:.0f} hours {m:.0f} minutes and {s:.06f} seconds"
                    )
                    url = f"{self.api_url}/host/login.json"
                    login_data = {"token": tkn}
                    try:
                        response = self.session.post(url, data=login_data)
                        response.raise_for_status()

                    except HTTPError as err:
                        logger.error(f"{lp} error while refreshing: {err}")
                        raise err
                    else:
                        resp_json = response.json()
                        logger.debug(f"{lp} got API refresh response: {resp_json}")
                        """
                        {
                            'access_token': '<TOKEN>', # JWT
                            'access_token_expires': 21600, # Seconds
                            'credentials': 'auth=abcCBA123321', # legacy auth?
                            'append_password': 0, # ?
                            'version': '1.37.27', # ZM Version
                            'apiversion': '2.0' # API Version
                        }
                        """
                        if resp_json:
                            self.api_version = resp_json.get("apiversion")
                            self.zm_version = resp_json.get("version")
                            if (
                                "access_token" in resp_json
                                and resp_json["access_token"]
                            ):
                                self.access_token = resp_json["access_token"]
                                tkn_exp = resp_json["access_token_expires"]
                                logger.debug(
                                    f"{lp} access token expires in {tkn_exp} seconds "
                                    f"on: {datetime.datetime.now() + datetime.timedelta(seconds=tkn_exp)}"
                                )
                                self.cached_tokens = None
                        else:
                            if reauth:
                                logger.error(
                                    f"{lp} no response from API refresh, trying again"
                                )
                                self._re_login(reauth=False)
                            else:
                                logger.error(
                                    f"{lp} no response from API refresh, trying again"
                                )

            if _login:
                logger.debug(f"{lp} calling login()")
                self._login()

    def _login(
        self, username: Optional[SecretStr] = None, password: Optional[SecretStr] = None
    ):
        lp: str = "api::login::"
        login_data: Dict = {}
        url = f"{self.api_url}/host/login.json"
        if not username:
            username = self.username
        if not password:
            password = self.password
        if username and password:
            logger.debug(
                f"{lp} Credentials have been supplied",
            )
            login_data = {
                "user": username.get_secret_value(),
                "pass": password.get_secret_value(),
            }
        else:
            logger.debug(
                f"{lp} not using auth (no username and/or password was supplied)"
            )
            url = f"{self.api_url}/host/getVersion.json"
        try:
            response = self.session.post(url, data=login_data)
            response.raise_for_status()

        except HTTPError as err:
            """401: unauthorized - user authentication can allow access to the resource.
            403: forbidden - re-authenticating makes no difference. The access is tied to the application logic,
            such as insufficient rights to a resource.
            404: not found - RESOURCE TEMP OR PERM GONE"""
            code_ = err.response.status_code
            err_msg = err.response.json()
            logger.error(
                f"{lp} got API login error -> Code: {code_}  || JSON: {err_msg}"
            )
            if code_ == 401:
                if err_msg:
                    """
                    BAD_USERNAME: {
                        "success":false,
                        "data":{
                            "name":"Could not retrieve user te details",
                            "message":"Could not retrieve user te details",
                            "url":"\/zm\/api\/host\/login.json",
                            "exception":{
                                "class":"UnauthorizedException",
                                "code":401,"message":"Could not retrieve user te details"
                            }
                        }
                    }
                    BAD_PASSWORD: {
                        "success":false,
                        "data":{
                            "name":"Login denied for user &quot;testapi&quot;",
                            "message":"Login denied for user &quot;testapi&quot;",
                            "url":"\/zm\/api\/host\/login.json",
                            "exception":{
                                "class":"UnauthorizedException",
                                "code":401,
                                "message":"Login denied for user \"testapi\""
                            }
                        }
                    }
                    """
                    if "name" in err_msg["data"]:
                        if "Could not retrieve user" in err_msg["data"]["name"]:
                            logger.error(f"{lp} invalid username")
                        elif "Login denied for user" in err_msg["data"]["name"]:
                            logger.error(f"{lp} invalid password")
            elif code_ == 521:
                logger.error(
                    f"{lp} CloudFlare reports that the origin server is down [{code_}]"
                )

            raise err

        self.access_token = None
        self.refresh_token = None
        resp_json = response.json()
        if resp_json:
            self.api_version = resp_json.get("apiversion")
            self.zm_version = resp_json.get("version")
            if "access_token" in resp_json and resp_json["access_token"]:
                self.access_token = resp_json["access_token"]

                access_token_expires = resp_json["access_token_expires"]
                logger.debug(
                    f"{lp} access token expires in {access_token_expires} seconds "
                    f"on: {datetime.datetime.now() + datetime.timedelta(seconds=access_token_expires)}"
                )

                if "refresh_token" in resp_json and resp_json["refresh_token"]:
                    self.refresh_token = resp_json["refresh_token"]
                    refresh_token_expires = resp_json["refresh_token_expires"]
                    logger.debug(
                        f"{lp} refresh token expires in {refresh_token_expires} seconds "
                        f"on: {datetime.datetime.now() + datetime.timedelta(seconds=refresh_token_expires)}",
                    )
                if self.access_token and self.refresh_token:
                    self.cached_tokens = None
            else:
                logger.debug(
                    f"{lp} it is assumed 'OPT_USE_AUTH' is disabled as no access token was returned"
                )

    @property
    def cached_tokens(self):
        data = None
        if self.token_file and self.token_file.exists():
            try:
                with self.token_file.open("rb") as f:
                    data = pickle.load(f)
            except Exception as err:
                logger.error(f"failed to load cached tokens from disk: {err}")
            else:
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
            finally:
                return data

    @cached_tokens.setter
    def cached_tokens(self, tokens: Optional[Dict[str, str]] = None):
        if not tokens:
            logger.debug(
                f"cached_tokens setter called with {tokens=}, using self.X_token"
            )
            tokens = {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
            }
        try:
            self.token_file.touch(exist_ok=True, mode=0o640)
            with self.token_file.open("wb") as f:
                pickle.dump(
                    tokens,
                    f,
                )
        except Exception as err:
            logger.error(f"{lp} error writing tokens to disk: {err}")
        else:
            logger.debug(f"{lp} tokens written to disk")

    @cached_tokens.deleter
    def cached_tokens(self):
        if self.token_file and self.token_file.exists():
            try:
                self.token_file.unlink()
            except Exception as err:
                logger.error(f"{lp} error deleting tokens from disk: {err}")

    async def get_all_event_data(
        self,
        event_id: int,
    ):
        """Returns the data from an 'Event' API call.
            ZoneMinder returns 3 structures in the JSON response.
        - Monitor data - A dict containing data about the event' monitor.
        - Event data - A dict containing all info about the current event.
        - Frame data - A list whose length is the current amount of frames in the frame buffer for the event, also contains data about the frames.

        :param event_id: (int) the event ID to query.
        """
        lp: str = "api::get_all_event_data::"
        _start = time.perf_counter()
        event: Optional[Dict] = None
        monitor: Optional[Dict] = None
        frame: Optional[List] = None
        event_tot_frames: Union[int, float, None] = None
        events_url = f"{self.api_url}/events/{event_id}.json"
        api_event_response = await self.make_async_request(url=events_url)
        event = api_event_response.get("event", {}).get("Event")
        monitor = api_event_response.get("event", {}).get("Monitor")
        frame = api_event_response.get("event", {}).get("Frame")
        event_tot_frames = len(frame)
        logger.debug(
            f"{lp} got event data in {time.perf_counter() - _start:.04f} seconds",
        )
        return event, monitor, frame, event_tot_frames

    @property
    def portal_base_url(self) -> str:
        """Returns the base URL for the ZoneMinder portal.

        Returns:
            str: the base URL for the ZoneMinder portal.
        """
        return self.portal_url

    async def get_monitor_data(
        self,
        mon_id: int,
    ):
        """Returns the data from a 'Monitor' API call."""
        lp: str = "api::get_monitor_data::"

        monitor: Optional[Dict] = None
        monitor_url = f"{self.api_url}/monitors/{mon_id}.json"
        try:
            api_monitor_response = await self.make_async_request(url=monitor_url)
        except Exception as e:
            logger.error(f"{lp} Error during Event data retrieval: {str(e)}")
            logger.debug(f"{lp} EXCEPTION>>> {e}")
        else:
            monitor = api_monitor_response.get("monitor", {}).get("Monitor")
        finally:
            # logger.debug(f" DEBUG Monitor data: {monitor}", caller=caller)
            return monitor

    def get_image(self, fid: Union[int, str]):
        pass

    async def make_async_request(
        self,
        url: Optional[str] = None,
        query: Optional[Dict] = None,
        payload: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        type_action: str = "get",
        use_creds: bool = False,
        zms_req: bool = False,
        re_auth: bool = True,
        quiet: bool = False,
        timeout: int = 15,
    ):
        lp: str = f"api::async {type_action}::"
        if not self.async_session:

            self.async_session = aiohttp.ClientSession()
        ssl = self.config.ssl_verify
        if ssl is not False:
            ssl = None

        async def parse_response(resp: aiohttp.ClientResponse):
            """Parses the response from the API call."""
            from aiohttp import client_exceptions

            resp_status = resp.status
            boundary: Optional[str] = None
            images = []

            try:
                _resp: Union[bytes, str, Dict, List, None] = None
                nph_headers: Union[Dict, bytes, None] = None
                resp.raise_for_status()
                # chunk iterating logic here to mitigate timeouts
                content_type = resp.headers.get("content-type")
                content_length = resp.headers.get("content-length")
                if content_length is not None:
                    content_length = int(content_length)
                cloudflare = resp.headers.get("Server", "").startswith("cloudflare")
                transfer_encoding = resp.headers.get("Transfer-Encoding", "chunked" if not content_length else None)

                if content_type.startswith("application/json"):
                    # JSON data
                    _resp = await resp.json()

                elif content_type.startswith("text/plain"):
                    # text data
                    _resp = await resp.text()

                elif content_type.startswith("multipart/x-mixed-replace"):
                    # logger.debug(
                    #     f"{lp} Content-Type='{content_type}' (ZMS CGI?)"
                    # )

                    if "boundary=" in content_type:
                        boundary = f"{content_type.split('boundary=')[1]}".encode()
                        # logger.debug(
                        #     f"{lp} boundary found in content-type header: {boundary}"
                        # )
                        # RFC calls for a leading '--' on the boundary
                        boundary = b"--" + boundary

                    if transfer_encoding == "chunked":
                        logger.debug(
                            f"{lp} iterating chunks"
                        )
                        chunk_size = 1024
                        _resp = b""
                        _begin = False
                        i = 0

                        # ZMS mode=jpeg sends a stream of images, we only want the fid we asked for
                        # ZM interprets that as the starting frame of the stream.

                        async for chunk in resp.content.iter_chunked(chunk_size):
                            i += 1
                            if boundary and boundary in chunk:
                                if _begin is False:
                                    _begin = True
                                    # strip out the first boundary
                                    _resp += chunk.split(boundary + b"\r\n")[1]
                                    continue
                                else:
                                    logger.debug(
                                        f"{lp} boundary found in chunk (size: {chunk_size}) #{i}, "
                                        f"breaking out of stream loop"
                                    )
                                    _resp += chunk.split(b"\r\n" + boundary)[0]
                                    _begin = False
                                    break

                            _resp += chunk
                    else:
                        logger.debug(f"DBG FOR ZMS >>>>> non chunked response! doing a read()")
                        _resp = await resp.read()
                    # split out nph headers from response
                    # example_response = b"Content-Type: image/jpeg\r\nContent-Length: 373781\r\n\r\n\xff\xd8\xff\xe0"

                    if _resp:
                        # remove boundary from response
                        if boundary:
                            if _resp.startswith(boundary):
                                _resp = _resp.split(boundary + b"\r\n")[1]
                        # b'--ZoneMinderFrame\r\nContent-Typ
                        #   e: image/jpeg\r\nContent-Length: 134116\r\n\r\n\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\n\
                        #   x07\x07\x08'
                        if _resp.startswith(b'Content-Type: image/jpeg'):
                            split_resp = _resp.split(b"\r\n\r\n")
                            # logger.debug(f"{lp} stripping out nph headers from response - {len(split_resp) = }")
                            if len(split_resp)>= 2:
                                nph_headers, _resp = split_resp[:2]
                                if nph_headers:
                                    nph_headers = {
                                        x.decode().split(": ")[0]: x.decode().split(": ")[1]
                                        for x in nph_headers.split(b"\r\n")
                                        if x
                                    }
                                    if (
                                        "Content-Type" in nph_headers
                                        and nph_headers["Content-Type"]
                                    ):
                                        content_type = nph_headers["Content-Type"]

                                    if "Content-Length" in nph_headers:
                                        content_length = int(nph_headers["Content-Length"])
                    else:
                        # logger.warning(f"{lp} no response from ZMS CGI? {_resp = }")
                        pass

                else:
                    _resp = await resp.read()

            except aiohttp.ClientResponseError as err:
                if resp_status == 401:
                    if self.access_token:
                        logger.error(
                            f"{lp} 401 Unauthorized, attempting to re-authenticate"
                        )
                        self._login()
                        logger.debug(
                            f"{lp} re-authentication complete, retrying async request"
                        )
                        return await self.make_async_request(
                            url=url,
                            query=query,
                            payload=payload,
                            type_action=type_action,
                            re_auth=False,
                        )
                    else:
                        raise err
                elif resp_status == 404:
                    # split the URL to check if 'token=' 'user(name)=' or 'pass(word)=' are in the URL
                    logger.warning(f"{lp} Got 404 (Not Found)")
                    # ZM returns 404 when an image cannot be decoded or the requested event does not exist
                else:
                    logger.debug(
                        f"{lp} NOT 200|401|404 SOOOOOOOOOOOOOOOO Code={resp_status} error: {err}"
                    )

            except asyncio.TimeoutError as err:
                logger.error(f"{lp} asyncio.TimeoutError: {err}", exc_info=True)

            except Exception as err:
                logger.error(f"{lp} Generic Exception: {err}", exc_info=True)

            else:

                # logger.debug(
                #     f"{lp} RESPONSE RECEIVED>>> CloudFlare={cloudflare} | {content_type=} | {content_length=}"
                #     f"\n\n HEADERS = {resp.headers}"
                # )

                if content_length is not None:
                    if content_length > 0:
                        if isinstance(_resp, str):
                            if _resp.casefold().startswith("no frame found"):
                                #  r.text = 'No Frame found for event(69129) and frame id(280)']
                                logger.warning(
                                    f"{lp} Frame was not found by ZM API! >>> {resp.text}"
                                )
                            else:
                                logger.debug(
                                    f"{lp} raising RE_LOGIN ValueError -> Non 0 byte response: {resp.text}"
                                )
                                raise ValueError("RE_LOGIN")
                    elif content_length <= 0:
                        # ZM returns 0 byte body if index not found (cant find frame ID/out of bounds)
                        logger.debug(
                            f"{lp} 'content-length' = {content_length} - WAS THIS AN IMAGE REQUEST? "
                            f"ZM cant find frame ID?"
                        )
                return _resp

        if payload is None:
            payload = {}
        if query is None:
            query = {}
        if headers is None:
            headers = {}

        type_action = type_action.casefold()
        if self.access_token:
            if not use_creds:
                query["token"] = self.access_token
            else:
                query["user"] = self.username
                query["pass"] = self.password

        show_url: str = (
            url.replace(str(self.portal_base_url), self.sanitize_str)
            if self.sanitize
            else url
        )
        show_payload: Union[Dict, str] = {}
        show_query: Union[Dict, str] = {}
        show_headers: Union[Dict, str] = {}
        if query:
            show_query = {
                k: f"{v[:20]}...{self.sanitize_str}"
                if self.sanitize and k == "token"
                else v
                for k, v in query.items()
            }
            show_query = f"query={show_query}"
        if payload:
            show_payload = {
                k: f"{v[:20]}...{self.sanitize_str}"
                if self.sanitize and k == "token"
                else v
                for k, v in payload.items()
            }
            show_payload = f" payload={show_payload}"

        if headers:
            show_headers = {
                k: f"{v[:20]}...{self.sanitize_str}"
                if self.sanitize and k.startswith("CF-Access-Client")
                else v
                for k, v in headers.items()
            }
            show_headers = f" headers={show_headers}"

        # logger.debug(
        #     f"{lp} {show_url} {show_payload if show_payload else ''} "
        #     f"{show_query if show_query else ''} "
        #     f"{show_headers if show_headers else ''}".rstrip()
        # ) if not quiet else None

        r: Optional[aiohttp.ClientResponse] = None
        if type_action == "get":
            async with self.async_session.get(
                url,
                params=query,
                ssl=ssl,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as r:
                return await parse_response(r)
        elif type_action == "post":
            async with self.async_session.post(
                url,
                data=payload,
                params=query,
                ssl=ssl,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as r:
                return await parse_response(r)
        elif type_action == "put":
            async with self.async_session.put(
                url,
                data=payload,
                params=query,
                ssl=ssl,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as r:
                return await parse_response(r)
        elif type_action == "delete":
            async with self.async_session.delete(
                url,
                data=payload,
                params=query,
                ssl=ssl,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as r:
                return await parse_response(r)
        else:
            raise TypeError(f"Unsupported request type: {type_action}")

    def make_request(
        self,
        url: Optional[str] = None,
        query: Optional[Dict] = None,
        payload: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        type_action: str = "get",
        re_auth: bool = True,
        quiet: bool = False,
    ) -> Union[Dict, Response]:
        lp: str = f"api::{type_action}::"
        if payload is None:
            payload = {}
        if query is None:
            query = {}
        if headers is None:
            headers = {}
        config_headers: Dict = self.config.headers
        logger.debug(f"{lp} DEBUG>>> headers: {headers} --- {config_headers = }")
        type_action = type_action.casefold()
        if config_headers:
            headers.update(config_headers)

        logger.debug(f"{lp} DEBUG>>> headers: {headers}")
        show_tkn: str = ""
        if self.access_token:
            query["token"] = self.access_token
            show_tkn = (
                f"{self.access_token[:20]}...{self.sanitize_str}"
                if self.sanitize
                else self.access_token
            )

        show_url: str = (
            url.replace(str(self.portal_base_url), self.sanitize_str)
            if self.sanitize
            else url
        )
        show_payload: str = ""
        show_query: Union[str, Dict] = f"token: '{show_tkn}'"
        if not self.access_token:
            show_query = query
        if payload:
            show_payload = f" payload={payload}"
        logger.debug(
            f"{lp} {show_url}{show_payload} query={show_query}",
        ) if not quiet else None

        try:
            r: Response
            if type_action == "get":
                r = self.session.get(url, params=query, headers=headers, timeout=240)
            elif type_action == "post":
                r = self.session.post(url, data=payload, params=query, headers=headers)
            elif type_action == "put":
                r = self.session.put(url, data=payload, params=query, headers=headers)
            elif type_action == "delete":
                r = self.session.delete(
                    url, data=payload, params=query, headers=headers
                )
            else:
                logger.error(f"{lp} unsupported request type: {type_action}")
                raise ValueError(f"Unsupported request type: {type_action}")
            r.raise_for_status()
        except HTTPError as http_err:
            code = http_err.response.status_code
            if code == 401:
                logger.debug(
                    f"{lp} Got 401 (Unauthorized) -> {http_err.response.json()}"
                )
                raise ValueError("RE_LOGIN")
            elif code == 404:
                logger.warning(f"{lp} Got 404 (Not Found) -> {http_err}")
                logger.debug(f"{http_err.response}")
                # ZM returns 404 when an image cannot be decoded or the requested event does not exist
                try:
                    # If this is an event request there will be json data
                    # If it is an image request there will be no json data
                    err_json: Optional[dict] = http_err.response.json()
                except JSONDecodeError as e:
                    logger.debug(f"{lp} Error parsing response for JSON: {e}")
                    logger.warning(f"{lp} Image not found: {url}")
                else:
                    logger.error(f"{lp} 404 to JSON ERROR response >>> {err_json}")
                    if err_json.get("success") is False:
                        # get the reason instead of guessing
                        err_name = err_json.get("data").get("message")
                        err_message = err_json.get("data").get("message")
                        err_url = err_json.get("data").get("url")
                        if err_name == "Invalid event":
                            logger.warning(f"{lp} Event not found: {url}")
            else:
                logger.debug(
                    f"{lp} NOT 200|401|404 SOOOOOOOOOOOOOOOO HTTP [{code}] error: {http_err}"
                )
        # If RE_LOGIN is raised, it will be caught by the caller
        except ValueError as val_err:
            err_msg = str(val_err)
            if err_msg == "RE_LOGIN":
                if re_auth:
                    logger.debug(f"{lp} retrying login once")
                    self._refresh_tokens_if_needed()
                    logger.debug(f"{lp} retrying failed request again...")
                    return self.make_request(
                        url, query, payload, type_action, re_auth=False
                    )
            else:
                raise val_err
        else:
            # Empty response, e.g. to DELETE requests, can't be parsed to json
            # even if the content-type says it is application/json
            content_type = r.headers.get("content-type")
            content_length = r.headers.get("content-length")
            logger.debug(
                f"{lp} SUCCESS request method: '{type_action}' content-type: {content_type}, "
                f"content-length: {content_length} -------- headers: {r.headers}"
            )

            if content_type.startswith("application/json") and r.text:
                return r.json()
            elif content_type.startswith("image/"):
                # return raw image data
                return r
            else:
                logger.debug(f"{lp}  >>> {r.text = }")
                # A non 0 byte response will usually mean it's an image eid request that needs re-login
                if content_length:
                    if content_length != "0":
                        if r.text.lower().startswith("no frame found"):
                            #  r.text = 'No Frame found for event(69129) and frame id(280)']
                            logger.warning(
                                f"{lp} Frame was not found by API! >>> {r.text}"
                            )
                        else:
                            logger.debug(
                                f"{lp} raising RE_LOGIN ValueError -> Non 0 byte response: {r.text}"
                            )
                            raise ValueError("RE_LOGIN")
                    elif content_length == "0":
                        # ZM returns 0 byte body if index not found (no frame ID or out of bounds)
                        logger.debug(f"{lp} {content_length = } >>> {r.text = }")
                        logger.debug(
                            f"{lp} raising BAD_IMAGE ValueError -> Content-Length = {content_length}"
                        )
                        raise ValueError("BAD_IMAGE")

    async def clean_up(self):
        """Exit the API session."""
        lp = f"{LP}close::"
        self.session.close()
        if self.async_session:
            if self.async_session.closed:
                logger.debug(f"{lp} async session already closed")
            else:
                await self.async_session.close()
        logger.debug(f"{lp} sessions closed")
