from __future__ import annotations

import asyncio
import datetime
import logging
import random
import string
import time
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING, Tuple, Any, Dict, List, Generator

import aiohttp
import numpy as np

from ...Log import CLIENT_LOGGER_NAME

if TYPE_CHECKING:
    from ...Models.config import GlobalConfig, APIPullMethod, ZMSPullMethod


logger = logging.getLogger(CLIENT_LOGGER_NAME)
LP = "media:"
g: Optional[GlobalConfig] = None


class PipeLine:
    _event_data_recursions: int = 0
    options: Union[APIPullMethod, ZMSPullMethod, None] = None
    event_tot_frames: int = 0
    attempted_fids: List[Union[int, str]] = list()
    start_datetime: Optional[datetime] = None
    event_end_datetime: Optional[datetime] = None
    event_tot_seconds: Optional[float] = None
    increment_by: Optional[Union[int, float]] = None
    current_frame: int = 1

    @staticmethod
    def parse_response(response: bytes) -> Tuple[Optional[bytes], Optional[str]]:
        if response.startswith(b"\xff\xd8\xff"):
            logger.debug(f"Response is a JPEG formatted image!")
            return response, f"mid_{g.mid}_rand_{random.randint(0,1000)}.jpg"
        else:
            logger.debug(f"Response is not a JPEG formatted image!")
            return None, None

    async def get_event_data(self, msg: Optional[str] = None):
        """Calls global DB get event data"""
        if msg:
            logger.debug(f"{LP}read>event_data: {msg}")

        if not self.event_ended or not g.Frame:
            try:
                await g.db.get_all_event_data(g.eid)

            except Exception as e:
                logger.error(f"{LP} error grabbing event data from DB -> {e}")
                # recurse
                if self._event_data_recursions < 3:
                    self._event_data_recursions += 1
                    await self.get_event_data(msg="retrying to grab event data...")
                else:
                    logger.error(
                        f"{LP} max recursions reached trying to grab event data, aborting!"
                    )
                    raise RuntimeError(
                        f"{LP} max recursions reached trying to grab event data, aborting!"
                    )
            else:
                self.event_tot_frames = len(g.Frame)
                self.event_end_datetime = g.db.end_datetime_from_eid(g.eid)
                self.start_datetime = g.db.start_datetime_from_eid(g.eid)
                self.event_tot_seconds = g.Event.get(
                    "Length",
                    (self.event_end_datetime - self.start_datetime).total_seconds(),
                )
                self.increment_by = int(self.event_tot_frames / self.event_tot_seconds)
                logger.debug(
                    f"{LP} grabbed event data from ZoneMinder DB for event '{g.eid}' -- event total Frames: "
                    f"{self.event_tot_frames} -- EndDateTime: {self.event_end_datetime} -- StartDateTime: "
                    f"{self.start_datetime} - EventTotSec: {self.event_tot_seconds} - IncrementBy: {self.increment_by} "
                    f"- has event ended: {self.event_ended}"
                )
        else:
            logger.debug(f"{LP} event has ended, no need to grab event data")

    def __init__(self):
        global g
        from ...main import get_global_config

        g = get_global_config()

    @property
    def event_ended(self):
        if self.event_end_datetime:
            return True
        return False

    @property
    def frames_attempted(self):
        return len(self.attempted_fids) or 0

    def _process_frame(
        self,
        image: bytes = None,
        end: bool = False,
    ) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        """Process the frame, increment counters, and return the image if there is one"""
        lp = f"{LP}process_frame:"
        if image:
            img_name = f"mid_{g.mid}-{self.attempted_fids[-1]}.jpg" if not g.past_event else self.current_frame
            # (bytes, image_file_name)
            return (
                image,
                img_name,
            )
        if end:
            logger.error(f"{lp} end has been called, no more images to process!")
            return False, None
        if g.past_event:
            self.current_frame = self.current_frame + self.increment_by
            logger.debug(
                f"{lp} incrementing next frame ID to read by {self.increment_by} = {self.current_frame}"
            )
        return None, None

    async def generate_image(
        self,
    ) -> Generator[Tuple[Optional[Union[bytes, bool]], Optional[str]]]:
        """Generator to return images from the source"""
        while True:
            yield await self.get_image()

    async def get_image(self) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        pass


class APIImagePipeLine(PipeLine):
    """An image grabber that uses ZoneMinders API as its source"""

    from ...Models.config import APIPullMethod

    def __init__(
        self,
        options: APIPullMethod,
    ):
        self.last_snapshot_id: Optional[int] = None
        lp = f"{LP}API:init::"
        assert options, f"{lp} no stream options provided!"
        super().__init__()
        #  INIT START 
        self.options = options
        logger.debug(f"{lp} options: {self.options}")
        self.has_event_ended: str = ""
        self.max_attempts = options.attempts
        if g.past_event:
            logger.debug(f"{lp} this is a past event, max image grab attempts set to 1")
            self.max_attempts = 1
        self.max_attempts_delay = options.delay
        self.sbf: Optional[int] = self.options.sbf
        self.fps: Optional[int] = self.options.fps
        self.skip_frames_calc: int = 0
        self.event_end_datetime: str = ""

        # Alarm frame is always the first frame (the frame that kicked the event off)
        # pre count buffer length+1 for alarm frame
        self.current_frame = self.buffer_pre_count + 1
        # The pre- / post-buffers will give the absolute minimum number of frames to grab, assuming no event issues
        self.total_min_frames = int(
            (self.buffer_post_count + self.buffer_pre_count) / self.capture_fps
        )
        # We don't know how long an event will be so set an upper limit of at least
        # pre- + post-buffers calculated as seconds because we are pulling <X> FPS
        self.total_max_frames = self.options.max_frames

    async def get_image(self) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        ret_img_name: str = ""
        if self.frames_attempted >= self.total_max_frames:
            logger.error(
                f"max_frames ({self.total_max_frames}) has been reached, stopping!"
            )
            return False, None

        import aiohttp

        response: Optional[aiohttp.ClientResponse] = None
        lp = f"{LP}read:"
        if self.frames_attempted > 0:
            logger.debug(
                f"{lp} [{self.frames_processed}/{self.total_max_frames} frames processed: {self.attempted_fids}]"
            )
        else:
            logger.debug(f"{lp} processing first frame!")
            _msg = f"{lp} checking snapshot ids enabled, will check every {self.options.snapshot_frame_skip} frames"
            if g.past_event:
                _msg = (
                    f"{lp} this is a past event (not live), skipping snapshot "
                    f"id checks (snapshot frame ID will not change)"
                )
            logger.debug(_msg) if self.options.check_snapshots else None

        curr_snapshot = None
        if self.options.check_snapshots:
            # Only run every <x> frames, if it's a live event
            if (
                (not g.past_event)
                and (self.frames_processed > 0)
                and (self.frames_processed % self.options.snapshot_frame_skip == 0)
            ):
                await self.get_event_data(
                    msg=f"grabbing data for snapshot comparisons..."
                )
                if curr_snapshot := int(g.Event.get("MaxScoreFrameId", 0)):
                    if self.last_snapshot_id is not None:
                        if curr_snapshot != self.last_snapshot_id:
                            logger.debug(
                                f"{lp} current snapshot frame id is not the same as the last snapshot id "
                                f"CURR:{curr_snapshot} - PREV:{self.last_snapshot_id}, grabbing new snapshot image"
                            )
                            self.current_frame = curr_snapshot
                        else:
                            logger.debug(
                                f"{lp} current snapshot frame id is the same as the last snapshot id "
                                f"CURR:{curr_snapshot} - PREV:{self.last_snapshot_id}, skipping frame"
                            )
                    self.last_snapshot_id = curr_snapshot
                else:
                    logger.warning(
                        f"{lp} Event: {g.eid} - No Snapshot Frame ID found in ZM API? -> {g.Event = }",
                    )

        #  Check if we have already processed this frame ID 
        if self.current_frame in self.attempted_fids:
            logger.debug(
                f"{lp} skipping Frame ID: '{self.current_frame}' as it has already been"
                f" processed for event {g.eid}"
            )
            return self._process_frame()
        logger.debug(f"Calculated Frame ID as: {self.current_frame}")
        portal_url = str(g.api.portal_base_url)
        if portal_url.endswith("/"):
            portal_url = portal_url[:-1]
        fid_url = (
            f"{portal_url}/index.php?view=image&eid={g.eid}&fid={self.current_frame}"
        )
        timeout = g.config.zoneminder.pull_method.api.timeout or 15

        for image_grab_attempt in range(self.max_attempts):
            image_grab_attempt += 1
            logger.debug(
                f"{lp} attempt #{image_grab_attempt}/{self.max_attempts} to grab image ID: {self.current_frame}"
            )
            _perf = time.time()
            api_response = await g.api.make_async_request(fid_url, timeout=timeout)
            logger.debug(
                f"perf:{lp} API request took {time.time() - _perf:.5f)} seconds"
            )
            if isinstance(api_response, bytes) and api_response.startswith(
                b"\xff\xd8\xff"
            ):
                logger.debug(f"ZM API returned a JPEG formatted image!")
                return self._process_frame(image=api_response)
            else:
                resp_msg = ""
                if api_response:
                    resp_msg = f" response code={api_response.status} - response={api_response}"
                else:
                    resp_msg = f" no response received!"

                logger.warning(f"{lp} image was not retrieved!{resp_msg}")

                await self.get_event_data(msg="checking if event has ended...")

                if self.event_ended:  # Assuming event has ended
                    logger.debug(f"{lp} event has ended, checking OOB status...")
                    # is current frame OOB
                    if self.current_frame > self.event_tot_frames:
                        # We are OOB, so we are done
                        logger.debug(
                            f"{lp} we are OOB in a FINISHED event (current requested fid: {self.current_frame} > "
                            f"total frames in event: {self.event_tot_frames})"
                        )
                        return self._process_frame(end=True)
                else:
                    logger.debug(
                        f"{lp} event has not ended yet! Total Frames: {self.event_tot_frames}"
                    )
                if not g.past_event and (image_grab_attempt < self.max_attempts):
                    logger.debug(f"{lp} sleeping for {self.options.delay} second(s)")
                    time.sleep(self.options.delay)

        return self._process_frame()

    @property
    def frames_processed(self):
        return len(self.attempted_fids) or 0


class ZMSImagePipeLine(PipeLine):
    """
    This image pipeline is designed to work with ZM CGI script nph-zms.
    nph = No Parsed Headers (the CGI script handles all response data, bypassing the server for efficiency)
    **nph-zms is symlinked to zms**

    http://localhost/zm/cgi-bin/nph-zms?mode=single&monitor=1&user=USERNAME&pass=PASSWORD"
    works with token='<ACCESS TOKEN>' as well
    mode=jpeg or single (single mode 'fixed' in ZM commit: https://github.com/ZoneMinder/zoneminder/commit/d968e243ff0079bae5c9eb4519c022bab7cbf5a9)
    monitor=<mid> will ask for monitor mode
    event=<eid> will ask for event mode
    frame=<fid> will ask for a specific frame from an event (event mode)
    """

    max_frames: Optional[int] = 0

    def __init__(
        self,
        options: ZMSPullMethod,
    ):
        self.live_frame: int = 0
        self.lp = f"{LP}ZMS:"
        lp = f"{self.lp}init::"
        assert options, f"{lp} no stream options provided!"
        super().__init__()
        #  INIT START 
        self.async_session = g.api.async_session
        self.options = options
        self.base_url: Optional[str] = str(options.url) if options.url else None
        self.built_url: Optional[str] = None
        logger.debug(f"{lp} options: {self.options}")

        self.max_attempts = options.attempts or 1
        self.max_attempts_delay = options.delay
        self.max_frames = options.max_frames
        # Process URL, if it is empty grab ZM portal and append ZM_PATH_CGI
        if not self.base_url:
            logger.debug(
                f"{lp} no URL provided, constructing from ZM portal (config file) and ZMS_CGI_PATH from zm.conf files"
            )
            cgi_sys_path = Path(g.db.cgi_path)
            # ZM_PATH_CGI=/usr/lib/zoneminder/cgi-bin
            portal_url = str(g.api.portal_base_url)
            if portal_url.endswith("/"):
                portal_url = portal_url[:-1]
            self.base_url = f"{portal_url}/{cgi_sys_path.name}/nph-zms"

        if g.past_event:
            # if this is a past (non-live) event, grab event data
            # use a task so we don't block the main thread
            loop = asyncio.get_running_loop()
            loop.create_task(self.get_event_data())


    async def _req_img(
        self,
        url: Optional[str] = None,
        timeout: int = 15,
    ):
        if not isinstance(g.api.async_session, aiohttp.ClientSession):
            self.async_session = g.api.async_session = aiohttp.ClientSession()
        assert isinstance(self.async_session, aiohttp.ClientSession), f"Invalid session type: {type(self.async_session)}"

        lp: str = self.lp
        verify_ssl = g.api.config.ssl_verify
        if verify_ssl is None:
            verify_ssl = False
        query = {}
        if g.api.access_token:
            query["token"] = g.api.access_token
        resp: Optional[aiohttp.ClientResponse] = None
        async with self.async_session.post(
            url,
            params=query,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp_status = resp.status
            iterated_resp: Optional[bytes] = None
            boundary: Optional[bytes] = None
            nph_headers: Union[Dict, bytes, None] = None
            logger.debug(f"{lp} response status: {resp_status} -- headers: {resp.headers}")
            try:
                resp.raise_for_status()
                # ZMS embeds its own headers, these are the actual headers from Apache
                # If mode = jpeg, ZMS will send a stream of images that begin with headers
                content_type = resp.headers.get("content-type")
                content_length = resp.headers.get("content-length")
                if content_length is not None:
                    content_length = int(content_length)
                transfer_encoding = resp.headers.get("Transfer-Encoding")
                if "multipart/x-mixed-replace" in content_type:
                    # ZMS mode=jpeg , get the boundary
                    if "boundary=" in content_type:
                        boundary = f"{content_type.split('boundary=')[1]}".encode()
                        # RFC calls for a leading '--' on the boundary
                        boundary = b"--" + boundary
                        img_type = ""
                        img_length = 0
                        chunk_size = 1024
                        iterated_resp = b""
                        _begin = False
                        i = 0
                        logger.debug(f"{lp} iterating chunks (size: {chunk_size}) as ZMS mode=jpeg sends a stream of "
                                     f"images that begin with headers (nph)")
                        async for chunk in resp.content.iter_chunked(chunk_size):
                            i += 1
                            if boundary and boundary in chunk:
                                if _begin is False:
                                    _begin = True
                                    # strip out the first boundary
                                    _raw_resp = chunk.split(boundary + b"\r\n")[1]
                                    # b'--ZoneMinderFrame\r\nContent-Type: image/jpeg\r\n
                                    # Content-Length: 134116\r\n\r\n
                                    # \xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00
                                    nph_headers, iterated_resp = _raw_resp.split(b"\r\n\r\n")
                                    nph_headers = {
                                        x.decode()
                                        .split(": ")[0]: x.decode()
                                        .split(": ")[1]
                                        for x in nph_headers.split(b"\r\n")
                                        if x
                                    }
                                    logger.debug(f"{lp} nph_headers = {nph_headers}")
                                    img_type = nph_headers.get("Content-Type")
                                    img_length = int(nph_headers.get("Content-Length"))
                                    if not iterated_resp:
                                        logger.warning(f"{lp} no data found after headers in first chunk!")
                                    continue
                                else:
                                    logger.debug(
                                        f"{lp} boundary found in chunk (size: {chunk_size}) #{i}, "
                                        f"breaking out of stream (mode=jpeg) reading loop..."
                                    )
                                    iterated_resp += chunk.split(b"\r\n" + boundary)[0]
                                    _begin = False
                                    break

                            iterated_resp += chunk
                    else:
                        logger.warning(f"{lp} no boundary found in content-type header! -> {content_type}")
                else:
                    logger.debug(f"{lp} reading non-multipart response...")
                    iterated_resp = await resp.read()

            except aiohttp.ClientResponseError as err:
                # ZM throw 403 when token is expired, use existing refresh logic.
                # todo: async login
                if resp_status in {401, 403}:
                    if g.api.access_token:
                        logger.error(f"{lp} {resp_status} {'Unauthorized' if resp_status == 401 else 'Forbidden'}, "
                                     f"attempting to re-authenticate")
                        g.api._login()
                        logger.debug(f"{lp} re-authentication complete, retrying async request")
                        return await self._req_img(url=self.built_url, timeout=timeout)
                    else:
                        logger.exception(f"{lp} {resp_status} {'Unauthorized' if resp_status == 401 else 'Forbidden'}, "
                                         f"no access token found?")
                elif resp_status == 404:
                    logger.warning(f"{lp} Got 404 (Not Found), are you sure the url is correct? -> {self.built_url}")
                else:
                    logger.warning(
                        f"{lp} NOT 200|401|403|404 - Code={resp_status} error: {err}"
                    )

            except asyncio.TimeoutError as err:
                logger.error(f"{lp} asyncio.TimeoutError: {err}", exc_info=True)

            except Exception as err:
                logger.error(f"{lp} Generic Exception: {err}", exc_info=True)

            else:
                if content_length is not None:
                    if content_length > 0:
                        if isinstance(iterated_resp, str):
                            if iterated_resp.casefold().startswith("no frame found"):
                                #  resp.text = 'No Frame found for event(69129) and frame id(280)']
                                logger.warning(
                                    f"{lp} Frame was not found by ZMS! >>> {resp.text}"
                                )
                return iterated_resp

    async def get_image(self) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        if self.frames_attempted >= self.max_frames:
            logger.error(f"max_frames ({self.max_frames}) has been reached, stopping!")
            return False, None

        lp = f"{LP}ZMS:read:"
        timeout = self.options.timeout or 15

        if not g.past_event:
            self.live_frame += 1
            # Live event - Pull live image from monitor
            self.built_url = f"{self.base_url}?mode=single&monitor={g.mid}"
            fid = f"live_{self.live_frame}"
            self.attempted_fids.append(fid)
            start_img_req = time.time()
            zms_response = await self._req_img(self.built_url, timeout)
            # zms_response = await g.api.make_async_request(
            #     url=url, type_action="post", timeout=timeout
            # )
            end_img_req = time.time() - start_img_req
            logger.debug(f"perf:{lp} ZMS request {self.built_url} took: {end_img_req:.5f}")
            no_img_reason = ""
            if not zms_response:
                no_img_reason = f" no response received!"
            elif isinstance(zms_response, bytes):
                if zms_response.startswith(b"\xff\xd8\xff"):
                    logger.debug(f"{lp} Response is a JPEG formatted image")
                    return zms_response, f"mid_{g.mid}-{fid}.jpg"
                else:
                    no_img_reason = f" Non JPEG bytes data returned -> {zms_response}"
            else:
                no_img_reason = f" response is not bytes -> Type: {type(zms_response)} -- {zms_response = }"
            logger.warning(f"{lp} image was not retrieved!{no_img_reason}")
            return None, None

        else:
            # single mode should work for past events now. Keep this stream logic for future use
            # Past event, use mode=jpeg, iterate event using frame id's (target 1 fps)
            if self.frames_attempted == 0:
                logger.debug(f"{lp} processing first frame!")

            self.built_url = f"{self.base_url}?mode=single&event={g.eid}&frame={self.current_frame}"
            past_perf = time.time()

            for image_grab_attempt in range(self.max_attempts):
                image_grab_attempt += 1
                logger.debug(
                    f"{lp} attempt #{image_grab_attempt}/{self.max_attempts} to grab past image from mid: "
                    f"{g.mid} event data"
                )
                zms_response = await self._req_img(
                    url=self.built_url, timeout=timeout
                )
                # logger.debug(f"{lp} URL: {url}")
                end_perf = time.time()
                logger.debug(f"perf:{lp} ZMS request {self.built_url} took: {end_perf - past_perf:.5f}")
                resp_msg = ""
                # Cover unset and None
                if not zms_response:
                    # if isinstance(api_response, aiohttp.ClientResponse):
                    #     resp_msg = f"<<response code={api_response.status}>> - response={api_response}"
                    resp_msg = f"no response received!"

                elif isinstance(zms_response, bytes):
                    if zms_response.startswith(b"\xff\xd8\xff"):
                        logger.debug(f"{lp} Response is a JPEG formatted image")
                        return self._process_frame(image=zms_response)
                else:
                    resp_msg = f"{lp} response is not bytes -> {type(zms_response) = } -- {zms_response = }"

                if resp_msg:
                    str_delay = ''
                    if self.max_attempts_delay:
                        str_delay = f" - sleeping for {self.max_attempts_delay} seconds"
                    logger.debug(
                        f"{lp} No image returned ({resp_msg}){str_delay}"
                    )
                    if self.max_attempts_delay:
                        await asyncio.sleep(self.max_attempts_delay)

                return self._process_frame()


# class SHMImagePipeLine(PipeLine):
#     class ZMAlarmStateChanges(IntEnum):
#         STATE_IDLE = 0
#         STATE_PREALARM = 1
#         STATE_ALARM = 2
#         STATE_ALERT = 3
#         STATE_TAPE = 4
#         ACTION_GET = 5
#         ACTION_SET = 6
#         ACTION_RELOAD = 7
#         ACTION_SUSPEND = 8
#         ACTION_RESUME = 9
#         TRIGGER_CANCEL = 10
#         TRIGGER_ON = 11
#         TRIGGER_OFF = 12
#
#     def __init__(self):
#         path = "/dev/shm"
#         # ascertain where the SHM filesystem is mounted
#         if not Path(path).exists():
#             path = "/run/shm"
#             if not Path(path).exists():
#                 raise FileNotFoundError(
#                     f"Cannot find SHM filesystem at /dev/shm or /run/shm"
#                 )
#
#         self.IS_64BITS = sys_maxsize > 2**32
#         # This will compensate for 32/64 bit
#         self.struct_time_stamp = r"l"
#         self.TIMEVAL_SIZE: int = struct.calcsize(self.struct_time_stamp)
#         self.alarm_state_stages = self.ZMAlarmStateChanges
#         self.file_handle: Optional[IO] = None
#         self.mem_handle: Optional[mmap.mmap] = None
#
#         self.file_name = f"{path}/zm.mmap.{g.mid}"
#
#     def reload(self):
#         """Reloads monitor information. Call after you get
#         an invalid memory report
#
#         Raises:
#             ValueError: if no monitor is provided
#         """
#         # close file handler
#         self.close()
#         # open file handler in read binary mode
#         self.file_handle = open(self.file_name, "r+b")
#         # geta rough size of the memory consumed by object (doesn't follow links or weak ref)
#         from os.path import getsize
#
#         sz = getsize(self.file_name)
#         if not sz:
#             raise ValueError(f"Invalid size: {sz} of {self.file_name}")
#
#         self.mem_handle = mmap.mmap(
#             self.file_handle.fileno(), 0, access=mmap.ACCESS_READ
#         )
#         self.sd = None
#         self.td = None
#         self.get_image()
#
#     def is_valid(self):
#         """True if the memory handle is valid
#
#         Returns:
#             bool: True if memory handle is valid
#         """
#         try:
#             d = self.get_image()
#             return not d["shared_data"]["size"] == 0
#         except Exception as e:
#             logger.debug(f"ERROR!!!! =-> Memory: {e}")
#             return False
#
#     def get_image(self):
#         self.mem_handle.seek(0)  # goto beginning of file
#         # import proper class that contains mmap data
#
#         sd_model = x.shared_data
#         td_model = x.trigger_data
#         vs_model = x.video_store_data
#
#         SharedData = sd_model.named_tuple
#
#         # old_shared_data = r"IIIIQIiiiiii????IIQQQ256s256s"
#         # I = uint32 - i = int32 == 4 bytes ; int
#         # Q = uint64 - q = int64 == 8 bytes ; long long int
#         # L = uint64 - l = int64 == 8 bytes ; long int
#         # d = double == 8 bytes ; float
#         # s = char[] == n bytes ; string
#         # B = uint8 - b = int8 == 1 byte; char
#
#         # shared data bytes is now aligned at 776 as of commit 590697b (1.37.19) -> SEE
#         # https://github.com/ZoneMinder/zoneminder/commit/590697bd807ab9a74d605122ef0be4a094db9605
#         # Before it was 776 for 64bit and 772 for 32 bit
#
#         TriggerData = td_model.named_tuple
#
#         VideoStoreData = vs_model.named_tuple
#
#         s = SharedData._make(
#             struct.unpack(sd_model.struct_str, self.mem_handle.read(sd_model.bytes))
#         )
#         t = TriggerData._make(
#             struct.unpack(td_model.struct_str, self.mem_handle.read(td_model.bytes))
#         )
#         v = VideoStoreData._make(
#             struct.unpack(vs_model.struct_str, self.mem_handle.read(vs_model.bytes))
#         )
#
#         written_images = s.last_write_index + 1
#         timestamp_offset = s.size + t.size + v.size
#         images_offset = timestamp_offset + g.mon_image_buffer_count * self.TIMEVAL_SIZE
#         # align to nearest 64 bytes
#         images_offset = images_offset + 64 - (images_offset % 64)
#         # images_offset = images_offset + s.imagesize * s.last_write_index
#         # Read timestamp data
#         ts_str = " "
#         if s.last_write_index <= g.mon_image_buffer_count >= 0:
#             for loop in range(written_images):
#                 ts_str = f"image{loop + 1}_ts"
#         self.mem_handle.seek(timestamp_offset)
#         TimeStampData = namedtuple(
#             "TimeStampData",
#             ts_str,
#         )
#         ts = TimeStampData._make(
#             struct.unpack(
#                 self.struct_time_stamp,
#                 self.mem_handle.read(g.mon_image_buffer_count * self.TIMEVAL_SIZE),
#             )
#         )
#         logger.debug(f"{written_images = } - {images_offset = }")
#         logger.debug(f"\nSharedData = {s}\n")
#         self.mem_handle.seek(images_offset)
#         # only need 1 image, not the recent buffers worth
#         image_buffer = self.mem_handle.read(int(s.imagesize))
#         # image_buffer = self.mem_handle.read(written_images * s.imagesize)
#         import cv2
#         import numpy as np
#
#         img: Optional[np.ndarray] = None
#         logger.debug(f"Converting images into ndarray")
#         # grab total image buffer
#         # convert bytes to numpy array to cv2 images
#         # for i in range(written_images):
#         # img = np.frombuffer(image_buffer[i * s.imagesize : (i + 1) * s.imagesize], dtype=np.uint8)
#         img = np.frombuffer(image_buffer, dtype=np.uint8)
#         if img.size == s.imagesize:
#             logger.debug(f"Image is of the correct size, reshaping and converting")
#             img = img.reshape((g.mon_height, g.mon_width, g.mon_colorspace))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         else:
#             logger.debug(f"Invalid image size: {img.size}")
#         # show images
#         self.sd = s._asdict()
#         self.td = t._asdict()
#         self.vsd = v._asdict()
#         self.tsd = ts._asdict()
#
#         self.sd["video_fifo_path"] = (
#             self.sd["video_fifo_path"].decode("utf-8").strip("\x00")
#         )
#         self.sd["audio_fifo_path"] = (
#             self.sd["audio_fifo_path"].decode("utf-8").strip("\x00")
#         )
#         self.sd["janus_pin"] = self.sd["janus_pin"].decode("utf-8").strip("\x00")
#         self.vsd["event_file"] = self.vsd["event_file"].decode("utf-8").strip("\x00")
#         self.td["trigger_text"] = self.td["trigger_text"].decode("utf-8").strip("\x00")
#         self.td["trigger_showtext"] = (
#             self.td["trigger_showtext"].decode("utf-8").strip("\x00")
#         )
#         self.td["trigger_cause"] = (
#             self.td["trigger_cause"].decode("utf-8").strip("\x00")
#         )
#         self.sd["alarm_cause"] = self.sd["alarm_cause"].decode("utf-8").strip("\x00")
#         self.sd["control_state"] = (
#             self.sd["control_state"].decode("utf-8").strip("\x00")
#         )
#
#         return {
#             "shared_data": self.sd,
#             "trigger_data": self.td,
#             "video_store_data": self.vsd,
#             "time_stamp_data": self.tsd,
#             "image": img,
#         }
#
#     def close(self):
#         """Closes the handle"""
#         try:
#             if self.mem_handle:
#                 self.mem_handle.close()
#             if self.file_handle:
#                 self.file_handle.close()
#         except Exception:
#             pass


class ZMUImagePipeLine(PipeLine):
    offset = None


class FileImagePipeLine(PipeLine):
    config: Optional[Dict[str, Any]] = None
    image: Optional[np.ndarray] = None
    input_file: Optional[Path] = None
    video: Optional[Any] = None


def zm_version(
    ver: str, minx: Optional[int] = None, patchx: Optional[int] = None
) -> int:
    maj, min, patch = "", "", ""
    x = ver.split(".")
    x_len = len(x)
    if x_len <= 2:
        maj, min = x
        patch = "0"
    elif x_len == 3:
        maj, min, patch = x
    else:
        logger.debug("come and fix me!?!?!")
    maj = int(maj)
    min = int(min)
    patch = int(patch)
    if minx:
        if minx > min:
            return 1
        elif minx == min:
            if patchx:
                if patchx > patch:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0
